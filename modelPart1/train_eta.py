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

from utils import init_logger, Timer, _AMP_NEW, set_seed, save_ckpt, load_ckpt, evaluate,collate_fn_eta
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
    (out/"config.json").write_text(json.dumps(vars(cfg), indent=2), encoding='utf-8')

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
        train_dl = DataLoader(train_ds, batch_size=cfg.batch,
                              shuffle=True, num_workers=cfg.workers,
                              collate_fn=collate_fn_eta, pin_memory=True)
        val_dl   = DataLoader(val_ds, batch_size=cfg.batch,
                              shuffle=False, num_workers=cfg.workers,
                              collate_fn=collate_fn_eta, pin_memory=True)

    # Model & optimizer
    emb = GroupEmbedder().to(device)
    mdl = ETAPredictorNet().to(device)
    params = list(emb.parameters()) + list(mdl.parameters())
    optim = Adam(params, lr=cfg.lr, weight_decay=cfg.wd)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs, eta_min=cfg.lr*0.1)
        if cfg.scheduler=='cosine'
        else torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=3)
    )
    criterion = MARELoss()
    scaler = GradScaler(enabled=cfg.amp)

    start_ep = 1
    if cfg.resume and Path(cfg.resume).exists():
        start_ep = load_ckpt(Path(cfg.resume), emb, mdl, optim, scaler, device) + 1
        print(f"🔄 Resume from epoch {start_ep}")

    best_val = float('inf')
    for ep in range(start_ep, cfg.epochs + 1):
        with Timer(f"Epoch {ep}"):
            run_loss = run_cnt = 0
            pbar = tqdm(train_dl, desc=f"E{ep}/{cfg.epochs} 0.000")
            for step, batch in enumerate(pbar):
                if 0 < cfg.max_batches <= step:
                    break
                batch = [x.to(device) for x in batch]
                (A_seq, A_stat,
                 near_seq, near_stat, dxy, dcs,
                 ship_seq, ship_stat,
                 B6, label) = batch
                B, K, Tn, _ = near_seq.shape
                H, Ts = ship_seq.shape[1], ship_seq.shape[2]

                ctx = autocast(device_type='cuda', enabled=cfg.amp) if _AMP_NEW else autocast(enabled=cfg.amp)
                with ctx:
                    # Embeddings
                    A_emb = emb(A_seq, A_stat)   # (B,128)
                    near_emb = emb(
                        near_seq.reshape(B*K, Tn, 7),
                        near_stat.reshape(B*K, 7)
                    ).view(B, K, -1)           # (B,K,128)
                    ship_emb = emb(
                        ship_seq.reshape(B*H, Ts, 7),
                        ship_stat.reshape(B*H, 7)
                    ).view(B, H, -1)           # (B,H,128)
                    # Predict ETA (hours)
                    pred = mdl(B6, A_emb, near_emb, dxy, dcs, ship_emb).squeeze(-1)  # (B,)
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
                pbar.set_description(f"E{ep}/{cfg.epochs} {run_loss/run_cnt:.3f}")

        # Validation
        with Timer("validation"):
            val_mare = evaluate(mdl, emb, val_dl, device, criterion, cfg.amp)
        if cfg.scheduler == 'plateau':
            scheduler.step(val_mare)
        else:
            scheduler.step()

        print(f"Epoch {ep}: Train MARE {run_loss/run_cnt:.3f}   Val MARE {val_mare:.3f}")

        # Save checkpoint
        ckpt = out/f"epoch_{ep}.pth"
        save_ckpt(ep, emb, mdl, optim, scaler, ckpt)
        if val_mare < best_val:
            best_val = val_mare
            shutil.copy(str(ckpt), str(out/"best.pth"))
            print(f"  ✔ New best (Val MARE {best_val:.3f})")

    print("🏁 Training complete. Best Val MARE:", best_val)

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument('--epochs',    type=int,   default=20)
    pa.add_argument('--batch',     type=int,   default=32)
    pa.add_argument('--max_batches', type=int, default=0,
                    help="Max number of batches per epoch (0 = no limit)")
    pa.add_argument('--lr',        type=float, default=2e-4)
    pa.add_argument('--wd',        type=float, default=0.0)
    pa.add_argument('--scheduler', choices=['cosine','plateau'], default='plateau')
    pa.add_argument('--k_near',    type=int,   default=40)
    pa.add_argument('--h_ship',    type=int,   default=10)
    pa.add_argument('--radius',    type=float, default=50.0)
    pa.add_argument('--step',      type=int,   default=1 ,help="Sampling step of the voyage(1=each node)")
    pa.add_argument('--clip',      type=float, default=0.0)
    pa.add_argument('--amp',       action='store_true', help="Use AMP")
    pa.add_argument('--seed',      type=int,   default=2025)
    pa.add_argument('--resume',    type=str,   default="")
    pa.add_argument('--out_dir',   type=str,   default="output")
    pa.add_argument('--log_level', type=str,   default="INFO")
    args = pa.parse_args()
    main(args)
