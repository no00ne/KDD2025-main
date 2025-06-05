#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py
───────────────────────────────────────────────────────────────────
• Use PgETADataset + ETAPredictorNet to train ETA prediction (hours).
• Supports:
    - AMP mixed precision
    - gradient clipping
    - Cosine/Plateau LR scheduler
    - checkpoint save/load
    - multi-process DataLoader
"""
import gc
import os
from functools import partial

from utils import eval_eta

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import math
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from torch.profiler import profile, ProfilerActivity, record_function
from utils import init_logger, Timer, _AMP_NEW, set_seed, save_ckpt, load_ckpt, collate_fn_eta
from pg_dataset_eta import PgETADataset
from eta_speed_model import GroupEmbedder, NewsEmbedder
from eta_eta_predictor import ETAPredictorNet


class MARELoss(nn.Module):
    """
    Mean Absolute Relative Error loss: mean(|(pred - target)/target|)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.mean(torch.abs((pred - target) / target))


def main(cfg):
    # Output directory
    init_logger(cfg.log_level)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Save config
    (out / "config.json").write_text(json.dumps(vars(cfg), indent=2), encoding='utf-8')

    set_seed(cfg.seed)
    device = torch.device(cfg.device if (cfg.device.startswith('cuda') and torch.cuda.is_available()) else 'cpu')
    cfg.amp = cfg.amp and device.type == 'cuda'
    print("▶ Device:", device, "| AMP:", cfg.amp)

    # Data
    train_ds = PgETADataset(True, cfg.k_near, cfg.h_ship, cfg.radius, cfg.step, m_news=cfg.m_news,
                            use_news=cfg.use_news)
    val_ds = PgETADataset(False, cfg.k_near, cfg.h_ship, cfg.radius, 10, m_news=cfg.m_news, use_news=cfg.use_news)

    collate = partial(collate_fn_eta, H=cfg.h_ship, K=cfg.k_near)

    with Timer("DataLoader build"):
        train_dl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.workers, pin_memory=True,
            collate_fn=collate)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg.workers, pin_memory=True,
            collate_fn=collate)

    # ---------- Model ----------

    Aemb = GroupEmbedder(use_news=cfg.use_news, m_news=cfg.m_news).to(device)  # 修改构造函数以接收新闻维度
    shipemb = GroupEmbedder(use_news=cfg.use_news, m_news=cfg.m_news).to(device)  # 修改构造函数以接收新闻维度
    nearemb = GroupEmbedder(use_news=cfg.use_news, m_news=cfg.m_news).to(device)  # 修改构造函数以接收新闻维度
    mdl = ETAPredictorNet(d_news=cfg.m_news, use_news=cfg.use_news).to(device)

    news_enc = None
    if cfg.use_news:
        news_enc = NewsEmbedder(d_in=16, d_out=128).to(device)
    params = list(Aemb.parameters()) + list(shipemb.parameters()) + list(nearemb.parameters()) + list(mdl.parameters())
    optim = Adam(params, lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs,
                                                            eta_min=cfg.lr * 0.1) if cfg.scheduler == 'cosine' else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.5, patience=3))
    criterion = MARELoss()
    scaler = GradScaler(enabled=cfg.amp)

    # ---------- (Optional) resume ----------
    start_ep = 1
    if cfg.resume and Path(cfg.resume).exists():
        start_ep = load_ckpt(
            Path(cfg.resume),
            Aemb,
            shipemb,
            nearemb,
            mdl,
            optim,
            scaler,
            device,
            strict=False,
        ) + 1
        # 允许重新指定学习率等可调参数
        for g in optim.param_groups:
            g['lr'] = cfg.lr
        print(f"🔄 Resume from epoch {start_ep} (lr={cfg.lr}, use_news={cfg.use_news})")

    # ---------- Train loop ----------
    best_val = float('inf')
    for ep in range(start_ep, cfg.epochs + 1):
        with Timer(f"Epoch {ep}"):
            run_loss = run_cnt = 0
            run_abs = 0.0
            run_sq = 0.0
            pbar = tqdm(train_dl, desc=f"E{ep}/{cfg.epochs} 0.000")

            # 在外面定义累积步数和一个 counter
            accum_steps = cfg.accum_steps  # 比如设置为 2、4 等
            optim.zero_grad()
            for step, batch in enumerate(pbar):
                if 0 < cfg.max_batches <= step:
                    break

                # ---------------- unpack 并 to(device) ----------------
                (A_raw, A_proj, A_len, A_stat, ship_raw, ship_proj, ship_len, ship_stat,
                 near_raw, near_proj, near_len, near_stat, dxy, dcs, dist_seg, speed_A, B6, label,
                 news_feat) = batch

                A_raw, A_proj, A_len, A_stat = A_raw.to(device), A_proj.to(device), A_len.to(device), A_stat.to(device)
                ship_raw, ship_proj, ship_len, ship_stat = ship_raw.to(device), ship_proj.to(device), ship_len.to(device), ship_stat.to(device)
                near_raw, near_proj, near_len, near_stat = near_raw.to(device), near_proj.to(device), near_len.to(device), near_stat.to(device)
                dxy, dcs, dist_seg = dxy.to(device), dcs.to(device), dist_seg.to(device)
                speed_A, B6, label = speed_A.to(device), B6.to(device), label.to(device)

                if cfg.use_news:
                    news_feat = news_feat[0].to(device)
                else:
                    news_feat = None

                B, nB, _ = near_raw.shape[:3]

                # ---------------- forward + backward ----------------
                ctx = autocast(device_type='cuda', enabled=cfg.amp) if _AMP_NEW else autocast(enabled=cfg.amp)
                with ctx:
                    # A 部分
                    A_seq_raw = A_raw.unsqueeze(1)  # (B, 1, T_A, 7)
                    A_seq_proj = A_proj.unsqueeze(2)  # (B, nB, 1, T_A, 7)
                    A_lengths = A_len.unsqueeze(1)  # (B, 1)
                    A_stat_exp = A_stat.unsqueeze(1)  # (B, 1, 16)
                    A_emb = Aemb(A_seq_raw, A_seq_proj, A_lengths, A_stat_exp)
                    # ship 部分
                    ship_seq_raw = ship_raw  # (B, H, T_ship, 7)
                    ship_seq_proj = ship_proj  # (B, nB, H, T_ship, 7)
                    ship_lengths = ship_len  # (B, H)
                    ship_stat_exp = ship_stat  # (B, H, 16)
                    ship_emb = shipemb(ship_seq_raw, ship_seq_proj, ship_lengths, ship_stat_exp)
                    # near 部分
                    near_emb = nearemb(near_raw, near_proj, near_len, near_stat)

                    # news
                    if news_feat is not None:
                        news_emb = news_enc(news_feat)
                    else:
                        news_emb = torch.zeros(B, nB, mdl.d_news, device=A_emb.device)

                    pred = mdl(B6, A_emb, near_emb, dxy, dcs, ship_emb, dist_seg, speed_A, news_emb).squeeze(-1)
                    loss = criterion(pred, label)
                    print(f'loss={loss}')
                    run_abs += torch.sum(torch.abs(pred.detach() - label)).item()
                    run_sq += torch.sum((pred.detach() - label) ** 2).item()
                    # —— 这里做梯度累积：先缩放 loss
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                # 只有当累积到指定步数时，才做一次 optimizer.step()
                if (step + 1) % accum_steps == 0:
                    if cfg.clip > 0:
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(params, cfg.clip)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

                # 累计统计 loss（注意这里累积前的 loss 已经被缩放过，因此乘回）

                run_loss += (loss.item() * accum_steps) * B
                run_cnt += B
                pbar.set_description(f"E{ep}/{cfg.epochs} {run_loss / run_cnt:.3f}")

                # -------- 释放显存 --------
                del A_raw, A_proj, A_len, A_stat
                del ship_raw, ship_proj, ship_len, ship_stat
                del near_raw, near_proj, near_len, near_stat
                del dxy, dcs, dist_seg, speed_A, B6, label
                del A_seq_raw, A_seq_proj, A_lengths, A_stat_exp
                del ship_seq_raw, ship_seq_proj, ship_lengths, ship_stat_exp
                del near_emb, ship_emb, A_emb, news_emb, pred, loss
                if news_feat is not None:
                    del news_feat
                gc.collect()
                torch.cuda.empty_cache()


        ckpt = out / f"epoch_{ep}.pth"
        save_ckpt(ep, Aemb, shipemb, nearemb, mdl, optim, scaler, ckpt)
        train_mare = run_loss / run_cnt
        train_mae = run_abs / run_cnt
        train_rmse = math.sqrt(run_sq / run_cnt)
        print(f"Epoch {ep}: Train MARE {train_mare:.3f} MAE {train_mae:.3f} RMSE {train_rmse:.3f}")
        # Validation（保持不变）
        with Timer("validation"):
            val_mare, val_mae, val_rmse = eval_eta(mdl, Aemb, shipemb, nearemb, val_dl, device, criterion, cfg.amp, news_enc)
        if cfg.scheduler == 'plateau':
            scheduler.step(val_mare)
        else:
            scheduler.step()

        print(
            f"Epoch {ep}: Train MARE {train_mare:.3f} MAE {train_mae:.3f} RMSE {train_rmse:.3f}   "
            f"Val MARE {val_mare:.3f} MAE {val_mae:.3f} RMSE {val_rmse:.3f}"
        )

        # Save checkpoint（保持不变）

        if val_mare < best_val:
            best_val = val_mare
            shutil.copy(str(ckpt), str(out / "best.pth"))
            print(f"  ✔ New best (Val MARE {best_val:.3f})")

    print("🏁 Training complete. Best Val MARE:", best_val)



if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument('--epochs', type=int, default=20)
    pa.add_argument('--batch', type=int, default=4)
    pa.add_argument('--max_batches', type=int, default=64, help="Max number of batches per epoch (0 = no limit)")
    pa.add_argument('--lr', type=float, default=1e-3)
    pa.add_argument('--wd', type=float, default=0.0)
    pa.add_argument('--scheduler', choices=['cosine', 'plateau'], default='plateau')
    pa.add_argument('--k_near', type=int, default=32)
    pa.add_argument('--h_ship', type=int, default=10)
    pa.add_argument('--radius', type=float, default=50.0)
    pa.add_argument('--step', type=int, default=16,
                    help="Maximum number of B nodes sampled from a voyage path")
    pa.add_argument('--clip', type=float, default=0.0)
    pa.add_argument('--amp', action='store_true', help="Use AMP")
    pa.add_argument('--seed', type=int, default=22)
    pa.add_argument('--resume', type=str, default="")
    pa.add_argument('--out_dir', type=str, default="output")
    pa.add_argument('--log_level', type=str, default="ERROR")
    pa.add_argument('--device', type=str, default='cuda:0',
                    help="PyTorch device, e.g. cuda:0 or cpu")
    # -------- 新增 --------
    pa.add_argument('--m_news', type=int, default=4, help="Dim of news feature vector; 0 = disable")
    pa.add_argument('--accum_steps', type=int, default=2)
    pa.add_argument('--use_news', action='store_true', help="Enable news feature flow when provided")

    # Dataloader workers
    pa.add_argument('--workers', type=int, default=8, help="num_workers for DataLoader")

    args = pa.parse_args()
    main(args)
