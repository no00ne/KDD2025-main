import logging
import math
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
import pandas as pd
from scipy.integrate import quad
from scipy.interpolate import CubicHermiteSpline

# ========== AMP 兼容导入 ==========
try:  # torch >= 2.3
    from torch.amp import autocast, GradScaler

    _AMP_NEW = True
except ImportError:  # torch 1.10 ~ 2.2
    from torch.cuda.amp import autocast, GradScaler

    _AMP_NEW = False


# -------------------- 随机 --------------------
def set_seed(seed=2025):
    random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)


# -------------------- Checkpoint --------------------
def save_ckpt(ep, emb, mdl, opt, scaler, path: Path):
    torch.save({
        'epoch': ep,
        'embedder': emb.state_dict(),
        'model': mdl.state_dict(),
        'optim': opt.state_dict(),
        'scaler': scaler.state_dict() if scaler else None
    }, str(path))


def load_ckpt(path: Path, emb, mdl, opt, scaler, device):
    ckpt = torch.load(str(path), map_location=device)
    emb.load_state_dict(ckpt['embedder'])
    mdl.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optim'])
    if scaler and ckpt.get('scaler'):
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', 0)


# -------------------- 计时 --------------------
def init_logger(level="INFO"):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO)
    )


@contextmanager
def Timer(name: str):
    logging.info(f"{name} ▶ start")
    t0 = time.perf_counter()
    yield
    logging.info(f"{name} ▶ end   (%.3f s)", time.perf_counter() - t0)


# -------------------- 编码工具 --------------------
def encode_time(ts):
    if not isinstance(ts, str):
        ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    dt = datetime.fromisoformat(ts)
    h = dt.hour + dt.minute / 60 + dt.second / 3600
    rad = 2 * math.pi * h / 24
    return math.sin(rad), math.cos(rad)


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance (km) between two points.
    """
    R = 6371.0
    φ1, φ2, Δφ, Δλ = map(math.radians, (lat1, lat2, lat2 - lat1, lon2 - lon1))
    a = math.sin(Δφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(Δλ / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def latlon_to_local(lat0, lon0, head, lat, lon, R_km=20.0):
    dx = haversine_km(lat0, lon0, lat0, lon) * math.copysign(1, lon - lon0)
    dy = haversine_km(lat0, lon0, lat, lon0) * math.copysign(1, lat - lat0)
    θ = math.radians(head)
    x_ = dx * math.cos(-θ) - dy * math.sin(-θ)
    y_ = dx * math.sin(-θ) + dy * math.cos(-θ)
    return x_ / R_km, y_ / R_km


# -------------------- 特征构造 (相对 B-ref 时间版) --------------------
def build_seq_tensor(nodes: list[dict], ref: dict) -> torch.Tensor:
    """
    生成局部投影特征张量 (T,7)，其中
      • ref 必须包含 'latitude', 'longitude', 'course', 'timestamp'
      • 时间维采用 “节点时间 − ref.timestamp” 的『24 h 相位差』编码

    字段顺序:
      0 speed_kmh           节点航速 (knots→km/h)
      1 sin(course_node)    节点航向正弦
      2 cos(course_node)    节点航向余弦
      3 dx_local            以 ref 为原点 / 船首方向的局部 x，单位: (dx/R) , R = 20 km
      4 dy_local            同上局部 y
      5 sin(Δt/24 h·2π)     Δt = node_ts - ref_ts (小时)
      6 cos(Δt/24 h·2π)
    """
    lat0, lon0 = ref["latitude"], ref["longitude"]
    head0 = ref.get("course", 0.0)  # 参考点船首朝向 (°)

    # 把 ref.timestamp 解析成 datetime
    ref_ts = ref.get("timestamp")
    if not isinstance(ref_ts, datetime):
        ref_ts = datetime.fromisoformat(str(ref_ts))

    feats = []
    for n in nodes:
        # --- ① 速度 & 航向 ---
        spd_kmh = (n.get("speed") or 0.0) * 1.852
        sin_c = math.sin(math.radians(n.get("course", 0.0)))
        cos_c = math.cos(math.radians(n.get("course", 0.0)))

        # --- ② 空间局部投影 ---
        dx, dy = latlon_to_local(
            lat0, lon0, head0,
            n["latitude"], n["longitude"]
        )  # 已除以 R_km=20 → 量级 [-1,1]

        # --- ③ 相对时间相位 (24 h) ---
        ts = n.get("timestamp")
        if not isinstance(ts, datetime):
            ts = datetime.fromisoformat(str(ts))
        delta_h = (ts - ref_ts).total_seconds() / 3600.0  # 可能为负
        phase = (delta_h % 24) / 24.0 * 2 * math.pi  # wrap 到 [0,24)
        sin_t = math.sin(phase)
        cos_t = math.cos(phase)

        feats.append([spd_kmh, sin_c, cos_c, dx, dy, sin_t, cos_t])

    return torch.tensor(feats, dtype=torch.float)


def project_seqs_to_B(
        raw_seqs: List[List[List[Dict]]],
        B_refs: List[Dict]
) -> List[List[List[torch.Tensor]]]:
    """
    对多组原始节点序列 raw_seqs，在每个 B_ref 处重新做 build_seq_tensor 投影。

    Args:
      raw_seqs: X 类，每类 Y 条记录，每条记录是一个节点 dict 列表，
                形如 [ [nodes_class0_item0, nodes_class0_item1, ...],   # 类 0, Y0 条
                        [nodes_class1_item0, ...],                     # 类 1, Y1 条
                        ... ]
      B_refs:   长度 n_B 的列表，每项是一个节点 dict（包含 latitude, longitude, course）

    Returns:
      result:   一个长度 n_B 的列表；
                result[b] 是一个 X×Y 的嵌套列表，其中
                  result[b][x][y] = build_seq_tensor(raw_seqs[x][y], B_refs[b])
    """
    from .utils import build_seq_tensor

    n_B = len(B_refs)
    X = len(raw_seqs)
    result: List[List[List[torch.Tensor]]] = []
    for b in range(n_B):
        ref = B_refs[b]
        per_B: List[List[torch.Tensor]] = []
        for x in range(X):
            per_class: List[torch.Tensor] = []
            for nodes in raw_seqs[x]:
                per_class.append(build_seq_tensor(nodes, ref))
            per_B.append(per_class)
        result.append(per_B)
    return result


def build_stat_tensor(v: dict):
    return torch.tensor([
        v.get("width", 0.0), v.get("length", 0.0), v.get("path_len", 0.0),
        v.get("start_lat", 0.0), v.get("start_lon", 0.0),
        v.get("end_lat", 0.0), v.get("end_lon", 0.0)
    ], dtype=torch.float)


# -------------------- padding --------------------
def _pad_inner(seqs: list[torch.Tensor]) -> torch.Tensor:
    """
    list[(T_i,7)] → (N, T_max, 7).  若全部空则返回 (N,1,7) 全零
    """
    if len(seqs) == 0: return torch.zeros(0, 1, 7)
    T = max(s.shape[0] for s in seqs) or 1
    out = torch.zeros(len(seqs), T, 7)
    for i, s in enumerate(seqs): out[i, :s.shape[0]] = s
    return out


def _pad_batch_time(tensors: list[torch.Tensor]) -> torch.Tensor:
    """
    list[(K,T_i,7)] → (B,K,T_max,7)
    """
    if len(tensors) == 0: return torch.zeros(0)
    K = tensors[0].shape[0]
    T = max(t.shape[1] for t in tensors) or 1
    out = torch.zeros(len(tensors), K, T, 7)
    for i, t in enumerate(tensors): out[i, :, :t.shape[1]] = t
    return out


def build_raw_seq_tensor(nodes: list[dict]) -> torch.Tensor:
    """
    对给定节点列表，提取原始特征：
      [speed(km/h), sin(course), cos(course), latitude, longitude, sin(hour), cos(hour)]
    并返回形状 (T, 7) 的张量。
    """
    feats = []
    for n in nodes:
        # 船速从节（knots）转到 km/h
        spd = (n.get("speed") or 0.0) * 1.852
        # 航向的 sin/cos
        sin_c = math.sin(math.radians(n.get("course", 0.0)))
        cos_c = math.cos(math.radians(n.get("course", 0.0)))
        # 时间编码
        sin_h, cos_h = encode_time(n.get("timestamp", ""))
        # 直接把纬度/经度当作原始特征
        feats.append([
            spd, sin_c, cos_c,
            n["latitude"], n["longitude"],
            sin_h, cos_h
        ])
    return torch.tensor(feats, dtype=torch.float)


def _pad(seqs: list[torch.Tensor]) -> torch.Tensor:
    """
    Pad list of [T_i, F] tensors to the same length in time dimension.
    Output shape [B, T_max, F].
    """
    T_max = max(s.shape[0] for s in seqs)
    B = len(seqs)
    F = seqs[0].shape[1]
    out = torch.zeros(B, T_max, F, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        out[i, :s.shape[0], :] = s
    return out


# -------------------- collate_fn --------------------
def collate_fn_speed(batch):
    # -------- A --------
    A_seq = _pad_inner([b["A_seq"] for b in batch])  # (B,T_A,7)
    A_len = torch.tensor([b["A_seq"].shape[0] for b in batch])
    A_stat = torch.stack([b["A_stat"] for b in batch])  # (B,7)

    # -------- Near ships --------
    near_seq = _pad_batch_time([_pad_inner(b["near_seqs"]) for b in batch])
    near_len = torch.tensor([[s.shape[0] for s in b["near_seqs"]] for b in batch])
    near_stat = torch.stack([torch.stack(b["near_stats"]) for b in batch])
    dxy = torch.stack([b["delta_xy"] for b in batch])
    dcs = torch.stack([b["delta_cs"] for b in batch])

    # -------- History --------
    ship_seq = _pad_batch_time([_pad_inner(b["ship_seqs"]) for b in batch])
    ship_len = torch.tensor([[s.shape[0] for s in b["ship_seqs"]] for b in batch])
    ship_stat = torch.stack([torch.stack(b["ship_stats"]) for b in batch])

    # -------- Point B features & label --------
    B6 = torch.stack([b["B_feat6"] for b in batch])
    label = torch.stack([b["label"] for b in batch])

    return (A_seq, A_len, A_stat,
            near_seq, near_len, near_stat, dxy, dcs,
            ship_seq, ship_len, ship_stat,
            B6, label)


# utils_pad.py ── 也可直接放到 utils 末尾 ────────────────────────────────
import torch
from typing import List

def _pad_seq(seq_list: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    list[(T_i,7)] → (N,T_max,7)  并返回 lengths:(N,)
    """
    if len(seq_list) == 0:
        return torch.zeros(0,1,7), torch.zeros(0,dtype=torch.long)

    T_max = max(s.size(0) for s in seq_list)
    out   = torch.zeros(len(seq_list), T_max, 7, dtype=seq_list[0].dtype)
    lens  = torch.zeros(len(seq_list), dtype=torch.long)
    for i,s in enumerate(seq_list):
        out[i,:s.size(0)] = s
        lens[i] = s.size(0)
    return out, lens


# ---------------------------------------------------------------
#  utils 辅助 – 两个新的 padding 例程
# ---------------------------------------------------------------
def _pad_2d_seqs(list2d):
    """
    list2d:  [N_i × (T,7)]  → (N_max, T_max, 7)   右补 0
    返回:  padded,  length_vec (N_max,)
    """
    if len(list2d) == 0:
        return torch.zeros(0, 1, 7), torch.zeros(0, dtype=torch.long)

    lengths = torch.tensor([s.shape[0] for s in list2d], dtype=torch.long)
    T_max   = int(lengths.max())
    N_max   = len(list2d)
    out     = torch.zeros(N_max, T_max, 7, dtype=list2d[0].dtype)

    for i, s in enumerate(list2d):
        out[i, : s.shape[0]] = s
    return out, lengths


def _pad_3d_nested(list3d, K, dim=7):
    """
    list3d:  len = n_B;   每项 list[K] → (T, dim) Tensor
    统一到 (n_B_max, K, T_max, dim)，并给 (n_B_max, K) 长度张量
    """
    n_B_max = len(list3d)
    # 统计整批最长序列
    T_max = 1
    for sub in list3d:
        for seq in sub:
            T_max = max(T_max, seq.shape[0])

    out  = torch.zeros(n_B_max, K, T_max, dim, dtype=list3d[0][0].dtype)
    lens = torch.zeros(n_B_max, K, dtype=torch.long)
    for i, sub in enumerate(list3d):          # sub: list[K]
        for j, seq in enumerate(sub):
            T = seq.shape[0]
            lens[i, j] = T
            out[i, j, :T] = seq
    return out, lens


# ---------------------------------------------------------------
#  collate_fn_eta
# ---------------------------------------------------------------
def collate_fn_eta(batch: list[dict]):
    """
    把 PgETADataset.__getitem__ 的 list[dict] 打包成一组张量 / 列表，供 DataLoader 使用。
    张量右侧补 0；有效步长分别放到 *_len 中。
    """
    B = len(batch)

    # ---------- A_raw / A_len ----------
    A_raw_pad, A_len = _pad_2d_seqs([b["A_raw"] for b in batch])           # (B,T_A_max,7)

    # ---------- A_proj ----------
    nB_max = max(len(b["A_proj_list"]) for b in batch)
    T_Amax = A_raw_pad.shape[1]
    A_proj_pad = torch.zeros(B, nB_max, T_Amax, 7)
    for i, b in enumerate(batch):
        for j, seq in enumerate(b["A_proj_list"]):
            A_proj_pad[i, j, : seq.shape[0]] = seq

    # ---------- A_stat ----------
    A_stat_pad = torch.stack([b["A_stat"] for b in batch])                 # (B,16)

    # ---------- ship ----------
    H = len(batch[0]["ship_raw_list"][0])          # 同一数据集固定 H
    ship_raw_nested  = []
    ship_proj_nested = []
    ship_len_nested  = []
    ship_stat_nested = []
    for b in batch:
        ship_raw_nested .append(b["ship_raw_list"])     # len n_B
        ship_proj_nested.append(b["ship_proj_list"])
        ship_stat_nested.append(b["ship_stats_list"])

    ship_raw_pad,  ship_len = _pad_3d_nested(ship_raw_nested,  H)          # (B,n_B,K=H,T,7)
    ship_proj_pad, _        = _pad_3d_nested(ship_proj_nested, H)
    # ship stats – (B,n_B,H,16)
    n_B_max = ship_raw_pad.shape[1]
    ship_stat_pad = torch.zeros(B, n_B_max, H, 16)
    for i, stat_B in enumerate(ship_stat_nested):
        for j, stat_list in enumerate(stat_B):
            for k, s in enumerate(stat_list):
                ship_stat_pad[i, j, k] = s

    # ---------- near ----------
    K = len(batch[0]["near_raw_list"][0])
    near_raw_nested  = []
    near_proj_nested = []
    near_stat_nested = []
    dxy_nested, dcs_nested = [], []
    for b in batch:
        near_raw_nested .append(b["near_raw_list"])
        near_proj_nested.append(b["near_proj_list"])
        near_stat_nested.append(b["near_stats_list"])
        dxy_nested.append(b["delta_xy"])
        dcs_nested.append(b["delta_cs"])

    near_raw_pad,  near_len = _pad_3d_nested(near_raw_nested,  K)          # (B,n_B,K,T,7)
    near_proj_pad, _        = _pad_3d_nested(near_proj_nested, K)
    # near stats – (B,n_B,K,16)
    near_stat_pad = torch.zeros(B, n_B_max, K, 16)
    for i, stat_B in enumerate(near_stat_nested):
        for j, stat_list in enumerate(stat_B):
            for k, s in enumerate(stat_list):
                near_stat_pad[i, j, k] = s

    # ---------- δxy / δcs ----------
    dxy_pad = torch.zeros(B, n_B_max, K, 2)
    dcs_pad = torch.zeros(B, n_B_max, K, 2)
    for i in range(B):
        dxy_pad[i, : batch[i]["delta_xy"].shape[0]] = batch[i]["delta_xy"]
        dcs_pad[i, : batch[i]["delta_cs"].shape[0]] = batch[i]["delta_cs"]

    # ---------- B6 ----------
    B6_pad = torch.zeros(B, n_B_max, 6)
    for i, b in enumerate(batch):
        B6_pad[i, : b["B6_list"].shape[0]] = b["B6_list"]

    # ---------- label ----------
    label_pad = torch.stack([b["label"] for b in batch])                   # (B,)

    # 输出 —— 可按训练循环需要调整顺序
    return (
        A_raw_pad,        A_proj_pad,        A_len,        A_stat_pad,
        ship_raw_pad,     ship_proj_pad,     ship_len,     ship_stat_pad,
        near_raw_pad,     near_proj_pad,     near_len,     near_stat_pad,
        dxy_pad,          dcs_pad,
        B6_pad,
        label_pad
    )




# -------------------- evaluate --------------------
@torch.no_grad()
def evaluate(model, embedder, loader, device, criterion, amp=True):
    model.eval();
    embedder.eval()
    tot, cnt = 0.0, 0
    for batch in loader:
        batch = [x.to(device) for x in batch]
        (A_seq, A_len, A_stat,
         near_seq, near_len, near_stat, dxy, dcs,
         ship_seq, ship_len, ship_stat,
         B6, label) = batch
        B, K, Tn, _ = near_seq.shape
        H, Ts = ship_seq.shape[1:3]
        ctx = autocast(device_type='cuda', enabled=amp) if _AMP_NEW else autocast(enabled=amp)
        with ctx:
            A_emb = embedder(A_seq, A_len, A_stat)
            near_emb = embedder(
                near_seq.reshape(B * K, Tn, 7),
                near_len.reshape(B * K),
                near_stat.reshape(B * K, 7)
            ).view(B, K, -1)
            ship_emb = embedder(
                ship_seq.reshape(B * H, Ts, 7),
                ship_len.reshape(B * H),
                ship_stat.reshape(B * H, 7)
            ).view(B, H, -1)
            pred = model(B6, A_emb, near_emb, dxy, dcs, ship_emb).squeeze(-1)
            loss = criterion(pred, label)
        tot += loss.item() * B;
        cnt += B
    model.train();
    embedder.train()
    return tot / cnt


def compute_hermite_distances(lats, lons, courses, R=6371000.0):
    """
    输入：
      lats:   (n,) 纬度序列（°）
      lons:   (n,) 经度序列（°）
      courses:(n,) 航向序列（°，0°=正北，顺时针）
      R:      地球半径，默认 6 371 000 m
    返回：
      (n-1,) 每段路程（m）
    """
    # 1) 投影到局部平面 (米)
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    lat0 = lat_rad[0]
    X = R * (lon_rad - lon_rad[0]) * np.cos(lat0)
    Y = R * (lat_rad - lat0)

    # 2) 参数 t：累积平面距离
    n = len(X)
    t = np.zeros(n)
    t[1:] = np.cumsum(np.hypot(np.diff(X), np.diff(Y)))

    # 3) 保证 t 严格递增
    mask = np.concatenate(([True], np.diff(t) > 0))
    t2 = t[mask]
    X2 = X[mask]
    Y2 = Y[mask]
    cr2 = np.deg2rad(courses[mask])  # 转成弧度

    # 4) 切向量：dx/dt, dy/dt 由航向给出(0°=北)
    #    北向对应 +Y，东向对应 +X
    dX2 = np.sin(cr2)
    dY2 = np.cos(cr2)

    # 5) 构造 Hermite 样条
    sx = CubicHermiteSpline(t2, X2, dX2)
    sy = CubicHermiteSpline(t2, Y2, dY2)

    # 6) 在每段 [t2[i], t2[i+1]] 上积分弧长
    dists = np.empty(len(t2) - 1)
    for i in range(len(dists)):
        a, b = t2[i], t2[i + 1]
        integrand = lambda u: np.hypot(sx(u, 1), sy(u, 1))
        dists[i], _ = quad(integrand, a, b)
    return dists


# -------- 第一次加载脚本时运行 --------
UN_PRED_NEWS = pd.read_csv("news_data/unpredictable.csv")
PRED_NEWS    = pd.read_csv("news_data/predictable.csv")

# 预先提取常用列为 numpy 数组，加快后续广播计算
cols = ["event_time", "north", "south", "east", "west"] + [f"score_{i}" for i in range(6)]
up_arr  = UN_PRED_NEWS[cols].values
pr_arr  = PRED_NEWS [cols].values

up_times = up_arr[:,0]
pr_times = pr_arr[:,0]
# ------------------------------------

def get_node_related_news_tensor(nodes, max_num=10, projection=True):
    """
    input:
      nodes: 节点列表, 每个节点是一个字典, 包含 timestamp, longitude, latitude 字段
      max_num: 每个节点最多选择的新闻数量
      projection: 是否使用 event_time - node_time 的时间差（投影）
    return:
      torch.Tensor shape (idx_of_nodes, idx_of_news, 8); 8: 2(event_time, delta_t) + 6(scores)
    """
    num_nodes = len(nodes)
    out = torch.zeros((num_nodes, max_num, 7), dtype=torch.float32)

    for idx, node in enumerate(nodes):
        t, lat, lon = node["timestamp"], node["latitude"], node["longitude"]

        # --- 时间切片（利用 searchsorted） ---
        up_cut = np.searchsorted(up_times, t, side="right")   # ≤ node_time
        pr_cut = np.searchsorted(pr_times, t, side="left")    #  > node_time
        slice_arr = np.vstack((up_arr[:up_cut], pr_arr[pr_cut:]))

        if slice_arr.size == 0:
            continue

        # ---------- 空间过滤 ----------
        lat_ok = (slice_arr[:, 1] >= lat) & (slice_arr[:, 2] <= lat)
        lon_ok = (slice_arr[:, 3] >= lon) & (slice_arr[:, 4] <= lon)
        hits   = slice_arr[lat_ok & lon_ok]

        if hits.shape[0]:
            hits = hits[:max_num]
            times    = hits[:, 0:1]
            delta_t  = times - t
            scores   = hits[:, 5:11]
            tensor   = torch.tensor(
                np.hstack((times, delta_t, scores)),
                dtype=torch.float32
            )

            out[idx, :tensor.shape[0], :] = tensor

    return out