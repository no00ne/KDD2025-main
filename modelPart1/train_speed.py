#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_speed.py
───────────────────────────────────────────────────────────────────
• 使用 PgETADataset + SpeedPredictor 训练 ETA 预测
• 支持:
    - AMP 混合精度
    - 梯度裁剪
    - Cosine/Plateau 学习率调度
    - checkpoint 保存/恢复
    - 多进程 DataLoader
"""
# 必须放在 import torch 之前
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from utils import init_logger, Timer, _AMP_NEW, collate_fn_eta
from utils                import set_seed, save_ckpt, load_ckpt, evaluate
from pg_dataset_speed           import PgETADataset, collate_fn_speed
from eta_speed_model      import GroupEmbedder, SpeedPredictor
from functools import partial

def main(cfg):
    # 输出目录
    init_logger(cfg.log_level)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # 保存 config
    (out/"config.json").write_text(
        json.dumps(vars(cfg), indent=2), encoding='utf-8')

    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.amp = cfg.amp and device.type=='cuda'
    print("▶ Device:", device, "| AMP:", cfg.amp)

    # Data
    train_ds = PgETADataset(True,  cfg.k_near, cfg.h_ship,
                             cfg.radius, cfg.step)
    val_ds   = PgETADataset(False, cfg.k_near, cfg.h_ship,
                             cfg.radius, cfg.step)
    with Timer("DataLoader build"):
        train_dl = DataLoader(
            train_ds,
            batch_size=cfg.batch,
            shuffle=True,
            num_workers=cfg.workers,
            pin_memory=True,
            collate_fn=partial(collate_fn_eta, cfg.h_ship, cfg.k_near)
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg.batch,
            shuffle=False,
            num_workers=cfg.workers,
            pin_memory=True,
            collate_fn=partial(collate_fn_eta, cfg.h_ship, cfg.k_near)
        )

    # Model & Opt
    emb = GroupEmbedder(use_news=false).to(device)
    mdl = SpeedPredictor().to(device)
    params = list(emb.parameters()) + list(mdl.parameters())
    optim  = Adam(params, lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=cfg.epochs, eta_min=cfg.lr*0.1
        ) if cfg.scheduler=='cosine'
        else torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.5, patience=3
        )
    )
    criterion = nn.L1Loss()
    scaler    = GradScaler('cuda',enabled=cfg.amp)

    start_ep = 1
    if cfg.resume and Path(cfg.resume).exists():
        start_ep = load_ckpt(Path(cfg.resume), emb, mdl,
                             optim, scaler, device) + 1
        print(f"🔄 Resume from epoch {start_ep}")

    best_val = float('inf')

    for ep in range(start_ep, cfg.epochs + 1):
        with Timer(f"Epoch {ep}"):
            run_loss = run_cnt = 0
            pbar = tqdm(train_dl, desc=f"E{ep}/{cfg.epochs} 0.000")
            for step, batch in enumerate(pbar):
                if 0 < cfg.max_batches <= step: break
                batch = [x.to(device) for x in batch]
                (A_seq, A_len, A_stat,
                 near_seq, near_len, near_stat, dxy, dcs,
                 ship_seq, ship_len, ship_stat,
                 B6, label) = batch
                B, K, Tn, _ = near_seq.shape
                H, Ts = ship_seq.shape[1:3]

                ctx = autocast(device_type='cuda', enabled=cfg.amp) if _AMP_NEW else autocast(enabled=cfg.amp)
                with ctx:
                    A_emb = emb(A_seq, A_len, A_stat)
                    near_emb = emb(
                        near_seq.reshape(B * K, Tn, 7),
                        near_len.reshape(B * K),
                        near_stat.reshape(B * K, 7)
                    ).view(B, K, -1)
                    ship_emb = emb(
                        ship_seq.reshape(B * H, Ts, 7),
                        ship_len.reshape(B * H),
                        ship_stat.reshape(B * H, 7)
                    ).view(B, H, -1)
                    pred = mdl(B6, A_emb, near_emb, dxy, dcs, ship_emb).squeeze(-1)
                    loss = criterion(pred, label)

                scaler.scale(loss).backward()
                if cfg.clip > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(params, cfg.clip)
                scaler.step(optim);
                scaler.update();
                optim.zero_grad()

                run_loss += loss.item() * B;
                run_cnt += B
                pbar.set_description(f"E{ep}/{cfg.epochs} {run_loss / run_cnt:.3f}")

        # ----- validation -----
        with Timer("validation"):
            val_mae = evaluate(mdl, emb, val_dl, device, criterion, cfg.amp)
        if cfg.scheduler == 'plateau':
            scheduler.step(val_mae)
        else:
            scheduler.step()

        # … 保存 checkpoint 部分保持 …
        print(f"Epoch {ep}: Train MAE {run_loss/run_cnt:.3f}   Val MAE {val_mae:.3f}")

            # 保存
        ckpt = out/f"epoch_{ep}.pth"
        save_ckpt(ep, emb, mdl, optim, scaler, ckpt)
        if val_mae < best_val:
            best_val = val_mae
            shutil.copy(str(ckpt), str(out/"best.pth"))
            print(f"  ✔ New best (Val MAE {best_val:.3f})")

    print("🏁 Training complete. Best Val MAE:", best_val)

if __name__ == "__main__":
    import json
    pa = argparse.ArgumentParser()
    pa.add_argument('--epochs',    type=int,   default=20)
    pa.add_argument('--batch',     type=int,   default=32)
    pa.add_argument('--max_batches', type=int, default=256,
                    help="每个 epoch 最多处理多少个 batch；0 = 不限制")
    pa.add_argument('--lr',        type=float, default=2e-4)
    pa.add_argument('--wd',        type=float, default=0.0)
    pa.add_argument('--scheduler', choices=['cosine','plateau'], default='plateau')
    pa.add_argument('--k_near',    type=int,   default=32)
    pa.add_argument('--h_ship',    type=int,   default=10)
    pa.add_argument('--radius',    type=float, default=50.0)
    pa.add_argument('--step',      type=int,   default=10)
    pa.add_argument('--workers',   type=int,   default=12)
    pa.add_argument('--amp',       action='store_true')
    pa.add_argument('--clip',      type=float, default=1.0,
                    help="梯度裁剪阈值，0 = 关闭")
    pa.add_argument('--log_level', type=str, default='INFO',
                    help="日志级别 DEBUG/INFO/WARNING/ERROR")

    pa.add_argument('--seed',      type=int,   default=2025)
    pa.add_argument('--out_dir',   type=str,   default='runs/eta_exp')
    pa.add_argument('--resume',    type=str,   default='',
                    help="路径到已有 checkpoint，以从该 epoch 继续")
    cfg = pa.parse_args()
    main(cfg)
