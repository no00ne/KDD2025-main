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
import os

from modelPart1.eta_speed_model import NewsEmbedder
from modelPart1.utils import eval_eta

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from utils import init_logger, Timer, _AMP_NEW, set_seed, save_ckpt, load_ckpt, evaluate, collate_fn_eta
from pg_dataset_eta import PgETADataset
from eta_speed_model import GroupEmbedder
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.amp = cfg.amp and device.type == 'cuda'
    print("▶ Device:", device, "| AMP:", cfg.amp)

    # Data
    train_ds = PgETADataset(True, cfg.k_near, cfg.h_ship,
                            cfg.radius, cfg.step,
                            m_news=cfg.m_news, use_news=cfg.use_news)
    val_ds = PgETADataset(False, cfg.k_near, cfg.h_ship,
                          cfg.radius, cfg.step,
                          m_news=cfg.m_news, use_news=cfg.use_news)

    with Timer("DataLoader build"):
        train_dl = DataLoader(train_ds, batch_size=cfg.batch,
                              shuffle=True, num_workers=cfg.workers,
                              collate_fn=collate_fn_eta, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=cfg.batch,
                            shuffle=False, num_workers=cfg.workers,
                            collate_fn=collate_fn_eta, pin_memory=True)

    # ---------- Model ----------
    emb = GroupEmbedder(cfg.m_news, cfg.use_news).to(device)  # 修改构造函数以接收新闻维度
    mdl = ETAPredictorNet(cfg.m_news, cfg.use_news).to(device)
    news_enc = NewsEmbedder(d_in=768, d_out=128).to(device)
    params = list(emb.parameters()) + list(mdl.parameters())
    optim = Adam(params, lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr * 0.1)
        if cfg.scheduler == 'cosine'
        else torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
    )
    criterion = MARELoss()
    scaler = GradScaler(enabled=cfg.amp)

    # ---------- (Optional) resume ----------
    start_ep = 1
    if cfg.resume and Path(cfg.resume).exists():
        start_ep = load_ckpt(Path(cfg.resume), emb, mdl, optim, scaler, device) + 1
        print(f"🔄 Resume from epoch {start_ep}")

    # ---------- Train loop ----------
    best_val = float('inf')
    for ep in range(start_ep, cfg.epochs + 1):
        with Timer(f"Epoch {ep}"):
            run_loss = run_cnt = 0
            pbar = tqdm(train_dl, desc=f"E{ep}/{cfg.epochs} 0.000")
            for step, batch in enumerate(pbar):
                if 0 < cfg.max_batches <= step:
                    break

                # ---------------- unpack ----------------
                (A_raw, A_proj, A_len, A_stat,
                 ship_raw, ship_proj, ship_len, ship_stat,
                 near_raw, near_proj, near_len, near_stat,
                 dxy, dcs, dist_seg,speed_A, B6, label,
                 news_feat) = batch

                A_raw = A_raw.to(device);
                A_proj = A_proj.to(device)
                A_len = A_len.to(device);
                A_stat = A_stat.to(device)
                ship_raw = ship_raw.to(device);
                ship_proj = ship_proj.to(device)
                ship_len = ship_len.to(device);
                ship_stat = ship_stat.to(device)
                near_raw = near_raw.to(device);
                near_proj = near_proj.to(device)
                near_len = near_len.to(device);
                near_stat = near_stat.to(device)
                dxy = dxy.to(device);
                dcs = dcs.to(device)
                dist_seg = dist_seg.to(device)
                speed_A = speed_A.to(device)
                B6 = B6.to(device);
                label = label.to(device)
                if cfg.use_news:
                    news_feat = news_feat[0].to(device)  # (B,nB,m_news) 或 (B,m_news)
                else:
                    news_feat = None

                B, nB, K = near_raw.shape[:3]
                H, Ts = ship_raw.shape[2:4]
                T_A, Tn = A_raw.size(1), near_raw.size(3)

                # ---------------- forward ----------------
                ctx = autocast(device_type='cuda', enabled=cfg.amp) if _AMP_NEW else autocast(enabled=cfg.amp)
                with ctx:
                    A_emb = emb(A_raw, A_proj, A_len, A_stat)  # (B,128)
                    near_emb = emb(
                        near_raw.reshape(B * nB * K, Tn, 7),
                        near_proj.reshape(B * nB * K, Tn, 7),
                        near_len.reshape(B * nB * K),
                        near_stat.reshape(B * nB * K, -1),
                    ).view(B, nB, K, -1)  # (B,nB,K,128)
                    # ---------- 2)  Ship-side ------------
                    # ❶ 仅取 **第 0 个 B_ref** 的 ship_raw / proj / stat 作为“基准”         ↓↓↓
                    ship_raw_base = ship_raw[:, 0]  # (B , H , Ts , 7)
                    ship_proj_base = ship_proj[:, 0]  # (B , H , Ts , 7)   ← 若只缓存 raw，可删掉
                    ship_len_base = ship_len[:, 0]  # (B , H)
                    ship_stat_base = ship_stat[:, 0]  # (B , H , 16)

                    # ❷ 摊平成一次前向：(B*H , Ts , 7) / (B*H , 16)
                    ship_emb_base = emb(
                        ship_raw_base.reshape(B * H, Ts, 7),
                        ship_proj_base.reshape(B * H, Ts, 7),  # 若不用 proj，这里传 None
                        ship_len_base.reshape(B * H),
                        ship_stat_base.reshape(B * H, -1),
                    ).view(B, H, -1)  # ➜ (B , H , 128)

                    # ❸ broadcast 到 nB 维——**不复制显存，只建 view**
                    ship_emb = ship_emb_base.unsqueeze(1).expand(-1, nB, -1, -1)  # (B , nB , H , 128)

                    if news_feat is not None:
                        news_emb = news_enc(news_feat).mean(dim=1)  # (B,128)
                    else:
                        news_emb = torch.zeros(A_emb.shape[0], 128, device=A_emb.device)
                    pred = mdl(B6, A_emb, near_emb, dxy, dcs,
                               ship_emb, dist_seg,speed_A,
                               news_emb).squeeze(-1)
                    loss = criterion(pred, label)

                scaler.scale(loss).backward()
                if cfg.clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

                run_loss += loss.item() * B
                run_cnt += B
                pbar.set_description(f"E{ep}/{cfg.epochs} {run_loss / run_cnt:.3f}")

        # Validation
        with Timer("validation"):
            val_mare = eval_eta(mdl, emb, val_dl, device, criterion, cfg.amp)
        if cfg.scheduler == 'plateau':
            scheduler.step(val_mare)
        else:
            scheduler.step()

        print(f"Epoch {ep}: Train MARE {run_loss / run_cnt:.3f}   Val MARE {val_mare:.3f}")

        # Save checkpoint
        ckpt = out / f"epoch_{ep}.pth"
        save_ckpt(ep, emb, mdl, optim, scaler, ckpt)
        if val_mare < best_val:
            best_val = val_mare
            shutil.copy(str(ckpt), str(out / "best.pth"))
            print(f"  ✔ New best (Val MARE {best_val:.3f})")

    print("🏁 Training complete. Best Val MARE:", best_val)


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument('--epochs', type=int, default=20)
    pa.add_argument('--batch', type=int, default=32)
    pa.add_argument('--max_batches', type=int, default=0,
                    help="Max number of batches per epoch (0 = no limit)")
    pa.add_argument('--lr', type=float, default=2e-4)
    pa.add_argument('--wd', type=float, default=0.0)
    pa.add_argument('--scheduler', choices=['cosine', 'plateau'], default='plateau')
    pa.add_argument('--k_near', type=int, default=40)
    pa.add_argument('--h_ship', type=int, default=10)
    pa.add_argument('--radius', type=float, default=50.0)
    pa.add_argument('--step', type=int, default=1, help="Sampling step of the voyage(1=each node)")
    pa.add_argument('--clip', type=float, default=0.0)
    pa.add_argument('--amp', action='store_true', help="Use AMP")
    pa.add_argument('--seed', type=int, default=2025)
    pa.add_argument('--resume', type=str, default="")
    pa.add_argument('--out_dir', type=str, default="output")
    pa.add_argument('--log_level', type=str, default="INFO")
    # -------- 新增 --------
    pa.add_argument('--m_news', type=int, default=0,
                    help="Dim of news feature vector; 0 = disable")
    pa.add_argument('--use_news', action='store_true',
                    help="Enable news feature flow when provided")

    # Dataloader workers
    pa.add_argument('--workers', type=int, default=8,
                    help="num_workers for DataLoader")

    args = pa.parse_args()
    main(args)
