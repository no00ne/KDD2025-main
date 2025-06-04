#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_modules.py  ──────────────────────────────────────────────
快速自检：
 1. utils 里常用函数
 2. PgETADataset.__getitem__（需 DB）
 3. 计时/输出完整字段
"""
import time, math, random, sys
from datetime import datetime, timedelta

import torch
import numpy as np

from utils import (
    encode_time,
    latlon_to_local,
    build_raw_seq_tensor,
    build_seq_tensor,
    collate_fn_eta,       # ← 需提前改好
    init_logger,
)

# Dataset 可能因 DB 不可用而导入失败
try:
    from pg_dataset_eta import PgETADataset
except Exception as e:
    PgETADataset = None
    print(f"[WARN] 无法导入 PgETADataset：{e}")

# ------- 1. utils --------
def test_utils():
    print("=== utils 简单验算 ===")
    sh, ch = encode_time("2025-05-29T12:00:00")
    print("encode_time(12:00) :", "%.3f %.3f" % (sh, ch))   # ≈  0 , -1

    dx, dy = latlon_to_local(10, 20,  30, 10.1, 20.1)       # 约 15 km => ≈ 0.75 @R=20
    print("latlon_to_local dx dy :", "%.3f %.3f" % (dx, dy))

    # 构造两节点，间隔 3h
    now   = datetime.utcnow()
    nodes = [
        dict(latitude=0, longitude=0, speed=10, course=0, timestamp=now.isoformat()),
        dict(latitude=0, longitude=0.1, speed=12, course=90,
             timestamp=(now+timedelta(hours=3)).isoformat()),
    ]
    raw  = build_raw_seq_tensor(nodes)
    proj = build_seq_tensor(nodes, nodes[0])     # 以第一个点为 ref

    # proj 第0行 sinΔt ≈0, cosΔt≈1; 第1行 Δt=3h => phase=3/24*2π
    print("build_seq_tensor first row time-phase :", proj[0,5:].tolist())
    print("build_raw_seq_tensor shape:", raw.shape)

    # fake batch 适配新版 collate_fn_eta
    fake_batch = [{
        "A_raw":            raw,
        "A_proj_list":      [proj],
        "A_main":           {"dummy":"ok"},
        "ship_raw_list":    [[raw]*2],
        "ship_stats_list":  [[{"x":1}]*2],
        "ship_proj_list":   [[proj]*2],
        "near_raw_list":    [[raw]*3],
        "near_stats_list":  [[{"y":2}]*3],
        "near_proj_list":   [[proj]*3],
        "delta_xy":         torch.randn(1,3,2),
        "delta_cs":         torch.randn(1,3,2),
        "B6_list":          torch.randn(1,6),
        "label":            torch.tensor(1.23),
    }]
    out = collate_fn_eta(fake_batch)
    print("collate_fn_eta 输出元素数 :", len(out))


# ------- 2. Dataset --------
def test_dataset():
    if PgETADataset is None:
        print("⚠️  跳过 Dataset 测试（导入失败或无数据库）")
        return
    print("\n=== PgETADataset 测试 ===")
    init_logger("INFO")

    ds = PgETADataset(train=True, k_near=32, h_ship=10,
                      radius_km=50.0, step=1,use_news=True,m_news=4)
    print("Dataset 长度 :", len(ds))

    t0 = time.time()
    sample = ds[random.randrange(len(ds))]
    print("getitem 用时 %.3f s" % (time.time()-t0))

    # 列出所有 key / 类型 / 结构
    for k,v in sample.items():
        if torch.is_tensor(v):
            print(f"  {k:<15} tensor {tuple(v.shape)}")
        elif isinstance(v, list):
            print(f"  {k:<15} list[{len(v)}]")
        elif isinstance(v, dict):
            print(f"  {k:<15} dict keys={list(v.keys())[:5]} ...")
        else:
            print(f"  {k:<15} {type(v)}")

# ---------------------------
if __name__ == "__main__":
    print("=== 开始测试 ===\n")

    test_dataset()
    print("\n=== 测试结束 ===")
