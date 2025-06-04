import logging
import math
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import warnings
logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
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
def save_ckpt(ep, Aemb, shipemb, nearemb, mdl, opt, scaler, path: Path):
    torch.save({
        'epoch': ep,
        'Aemb': Aemb.state_dict(),
        'shipemb': shipemb.state_dict(),
        'nearemb': nearemb.state_dict(),
        'model': mdl.state_dict(),
        'optim': opt.state_dict(),
        'scaler': scaler.state_dict() if scaler else None
    }, str(path))



def load_ckpt(path: Path,
              Aemb,
              shipemb,
              nearemb,
              mdl,
              opt=None,
              scaler=None,
              device=None,
              *,
              strict: bool = True):
    """Load checkpoint and restore module states.

    Parameters
    ----------
    strict : bool, optional
        When ``False`` mismatched keys in state dicts are ignored so that
        models with modified structures can still load available weights.
    """

    ckpt = torch.load(str(path), map_location=device)
    if Aemb is not None:
        Aemb.load_state_dict(ckpt['Aemb'], strict=strict)
    if shipemb is not None:
        shipemb.load_state_dict(ckpt['shipemb'], strict=strict)
    if nearemb is not None:
        nearemb.load_state_dict(ckpt['nearemb'], strict=strict)
    if mdl is not None:
        mdl.load_state_dict(ckpt['model'], strict=strict)
    if opt is not None and 'optim' in ckpt:
        opt.load_state_dict(ckpt['optim'])
    if scaler is not None and ckpt.get('scaler'):
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', 0)



# -------------------- 计时 --------------------
def init_logger(level="ERROR"):
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
        spd_kmh = (n.get("speed") or 0.0)
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
        spd = (n.get("speed") or 0.0)
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
# 文件：utils.py 中原来的 _pad_2d_seqs，整段替换为下面内容



def _pad_2d_seqs(list2d: List[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    list2d:  [Tensor_1, Tensor_2, ..., Tensor_N], 每个 Tensor_i 的形状是 (T_i, 7) 或可能是空张量 (0,)
    → 返回 padded: (N, T_max, 7)，后面补 0；lengths: (N,)
    """
    if len(list2d) == 0:
        return torch.zeros(0, 1, 7), torch.zeros(0, dtype=torch.long)

    # 首先将所有形如 (0,) 的“空”张量，转换成 (0, 7)
    cleaned = []
    for s in list2d:
        if s.ndim == 1 and s.numel() == 0:
            # 遇到 shape=(0,) 的情况，换成 shape=(0,7)
            cleaned.append(torch.zeros(0, 7, dtype=s.dtype))
        else:
            cleaned.append(s)

    # 计算每条序列的长度 T_i
    lengths = torch.tensor([s.shape[0] for s in cleaned], dtype=torch.long)
    T_max = int(lengths.max())      # 这一批里最长的时间维度
    N_max = len(cleaned)            # 批大小

    # 预分配输出张量，dtype 与第一个非空张量一致
    dtype = cleaned[0].dtype
    out = torch.zeros(N_max, T_max, 7, dtype=dtype)

    for i, s in enumerate(cleaned):
        # 对于“真正”有时间维的序列，直接复制；如果是 (0,7)，则 skip
        if s.shape[0] > 0:
            out[i, : s.shape[0]] = s

    return out, lengths



def _pad_3d_nested(list3d, K, dim=7):
    """
    list3d:  len = n_B;   每项 sub: list[K] → (T, dim) Tensor 或者是 嵌套列表 [Tensor, ...]
    统一到 (n_B, K, T_max, dim)，并给 (n_B, K) 长度张量
    """
    n_B_max = len(list3d)

    # 1. 第一遍：统计整批最长序列 T_max
    T_max = 1
    for sub in list3d:
        for seq in sub:
            if isinstance(seq, torch.Tensor):
                T_max = max(T_max, seq.shape[0])
            else:
                # seq 是列表时，取第一个 Tensor 来确定长度
                if len(seq) > 0 and isinstance(seq[0], torch.Tensor):
                    T_max = max(T_max, seq[0].shape[0])
                # 否则跳过

    # 2. 确定 dtype：直接探第一个元素，若还是列表则再探其第一个
    sample = list3d[0][0]
    if isinstance(sample, torch.Tensor):
        tensor_dtype = sample.dtype
    else:
        # sample 是 list，需要再取第一个 Tensor
        tensor_dtype = sample[0].dtype

    # 3. 分配输出张量和长度张量
    out = torch.zeros(n_B_max, K, T_max, dim, dtype=tensor_dtype)
    lens = torch.zeros(n_B_max, K, dtype=torch.long)

    # 4. 第二遍：逐位置把原始 seq 拷贝到 out，并记录真实长度
    for i, sub in enumerate(list3d):
        for j, seq in enumerate(sub):
            if isinstance(seq, torch.Tensor):
                T = seq.shape[0]
                lens[i, j] = T
                out[i, j, :T, :] = seq

            else:
                # seq 是列表，拿第一个 Tensor 来 pad
                if len(seq) > 0 and isinstance(seq[0], torch.Tensor):
                    first = seq[0]
                    T = first.shape[0]
                    lens[i, j] = T
                    out[i, j, :T, :] = first
                # 否则 lens 默认为 0，不做填充

    return out, lens



# ---------------------------------------------------------------
#  collate_fn_eta
# ---------------------------------------------------------------
import torch
from typing import List, Dict

def collate_fn_eta(batch: List[Dict], H: int, K: int):
    """
    batch:       List[dict]，每个 dict 对应 PgETADataset.__getitem__ 的输出。
    H:           同船轨迹个数上限
    K:           邻船轨迹个数上限

    返回（tuple），正好对应 train_eta.py 里 unpack 的 19 个输入维度：
      (A_raw_pad, A_proj_pad, A_len, A_stat_pad,
       ship_raw_pad, ship_proj_pad, ship_len, ship_stat_pad,
       near_raw_pad, near_proj_pad, near_len, near_stat_pad,
       dxy_pad, dcs_pad,
       dist_pad,
       speedA_pad,
       B6_pad,
       label_pad,
       news_pad)   # news_pad: (B, nB, M, 16) if not None
    """
    B = len(batch)

    # ========== A_raw / A_len ==========
    A_raw_pad, A_len = _pad_2d_seqs([b["A_raw"] for b in batch])  # (B, T_A_max, 7)

    # ========== A_proj ==========
    nB_max = max(len(b["A_proj_list"]) for b in batch)
    T_Amax = A_raw_pad.shape[1]
    A_proj_pad = torch.zeros(B, nB_max, T_Amax, 7)
    for i, b in enumerate(batch):
        for j, seq in enumerate(b["A_proj_list"]):
            A_proj_pad[i, j, : seq.shape[0]] = seq

    # ========== A_stat ==========
    A_stat_pad = torch.stack([b["A_stat"] for b in batch])  # (B,16)

    # ==================== ship 部分 ====================
    # 先对每个 (i,j) 内的 H 条序列在时间维度上 pad。如果某个 hist_list 少于 H 条，就补全 (1,7) 的零张量。
    ship_raw_tmp    = []   # ship_raw_tmp[i][0] 就是 “第 0 个 B_ref” 下的 (H, T_loc, 7)
    ship_proj_tmp   = []   # ship_proj_tmp[i][j]: (H, T_loc_j, 7)
    ship_len_local  = []   # ship_len_local[i][0] 是 “第 0 个 B_ref” 下的 H 个长度

    for b in batch:
        raw_per_sample  = []
        proj_per_sample = []
        len_per_sample  = []

        for idx_B, hist_list in enumerate(b["ship_raw_list"]):
            # 如果 hist_list 少于 H 条，就补成 H 条全零(1,7) Tensor
            if len(hist_list) < H:
                hist_list = hist_list + [torch.zeros(1, 7)] * (H - len(hist_list))
            # now len(hist_list) == H

            # pad 这 H 条轨迹
            stacked_raw, lengths_raw = _pad_2d_seqs(hist_list)  # (H, T_loc, 7)
            raw_per_sample.append(stacked_raw)

            # 对应的 proj_list 也补齐到 H 条
            proj_list = b["ship_proj_list"][idx_B]
            if len(proj_list) < H:
                proj_list = proj_list + [torch.zeros(1, 7)] * (H - len(proj_list))
            stacked_proj, lengths_proj = _pad_2d_seqs(proj_list)  # (H, T_loc, 7)
            proj_per_sample.append(stacked_proj)

            # lengths_raw 本身是 (H,) 向量，这里只取最大值表示 pad 后的 T_loc
            len_per_sample.append(int(lengths_raw.max().item()))

        ship_raw_tmp.append(raw_per_sample)
        ship_proj_tmp.append(proj_per_sample)
        ship_len_local.append(len_per_sample)

    # 只保留每个样本第 0 个 B_ref 下的 raw，输出为 (B, H, T_ship_pad, 7)
    n_B_max     = max(len(x) for x in ship_raw_tmp)
    T_ship_pad  = max(max(pl) for pl in ship_len_local)
    ship_raw_pad  = torch.zeros(B, H, T_ship_pad, 7)
    # 对齐 proj 维度为 (B, n_B_max, H, T_ship_pad, 7)
    ship_proj_pad = torch.zeros(B, n_B_max, H, T_ship_pad, 7)
    # ship_len: 只记录 “第 0 个 B_ref” 下的 H 个长度，形状 (B, H)
    ship_len      = torch.zeros(B, H, dtype=torch.long)
    # ship_stat_pad: 只保留 “第 0 个 B_ref” 下的 H 条 stat，形状 (B, H, 16)
    ship_stat_pad = torch.zeros(B, H, 16)

    for i, b in enumerate(batch):
        # 取第 0 个 B_ref 下 pad 后的 raw 张量，形状 (H, T_loc, 7)
        raw_0 = ship_raw_tmp[i][0]
        T_loc = raw_0.shape[1]
        ship_raw_pad[i, :, :T_loc, :] = raw_0

        # 对齐每个 j 下的 proj
        for j in range(n_B_max):
            if j < len(b["ship_proj_list"]):
                proj_j = ship_proj_tmp[i][j]  # (H, T_loc_j, 7)
                T_loc_j = proj_j.shape[1]
                ship_proj_pad[i, j, :, :T_loc_j, :] = proj_j

        # 记录 lengths：只取第 0 个 B_ref 下的 H 个长度
        ship_len[i, :] = torch.tensor(ship_len_local[i][0], dtype=torch.long)

        # 只填充第 0 个 B_ref 下的 H 条 stat
        for k, s in enumerate(b["ship_stats_list"][0]):
            ship_stat_pad[i, k] = s

    # ==================== near 部分 ====================
    # 对每个 (i,j) 内的 K 条轨迹 pad，如果某个 neigh_list 本身少于 K 条也补全零
    near_raw_tmp   = []
    near_proj_tmp  = []
    near_len_local = []  # near_len_local[i][j] 是一个 (K,) 向量

    for b in batch:
        raw_per_sample  = []
        proj_per_sample = []
        len_per_sample  = []

        for idx_B, neigh_list in enumerate(b["near_raw_list"]):
            # 如果 neigh_list 少于 K 条，就补成 K 条全零(1,7) Tensor
            if len(neigh_list) < K:
                neigh_list = neigh_list + [torch.zeros(1, 7)] * (K - len(neigh_list))
            stacked_raw, lengths_raw = _pad_2d_seqs(neigh_list)  # (K, T_loc, 7)
            raw_per_sample.append(stacked_raw)

            proj_list = b["near_proj_list"][idx_B]
            if len(proj_list) < K:
                proj_list = proj_list + [torch.zeros(1, 7)] * (K - len(proj_list))
            stacked_proj, lengths_proj = _pad_2d_seqs(proj_list)  # (K, T_loc, 7)
            proj_per_sample.append(stacked_proj)

            len_per_sample.append(lengths_raw)  # lengths_raw 是 (K,) 向量

        near_raw_tmp.append(raw_per_sample)
        near_proj_tmp.append(proj_per_sample)
        near_len_local.append(len_per_sample)

    T_near_pad   = max(max(lens.max().item() for lens in pl) for pl in near_len_local)
    # near_raw_pad： (B, n_B_max, K, T_near_pad, 7)
    near_raw_pad  = torch.zeros(B, n_B_max, K, T_near_pad, 7)
    near_proj_pad = torch.zeros(B, n_B_max, K, T_near_pad, 7)
    # near_len： (B, n_B_max, K)
    near_len      = torch.zeros(B, n_B_max, K, dtype=torch.long)
    # near_stat_pad: (B, n_B_max, K, 16)
    near_stat_pad = torch.zeros(B, n_B_max, K, 16)

    for i, b in enumerate(batch):
        for j, raw_stacked in enumerate(near_raw_tmp[i]):
            T_loc = raw_stacked.shape[1]
            near_raw_pad[i, j, :, :T_loc, :]  = raw_stacked
            near_proj_pad[i, j, :, :T_loc, :] = near_proj_tmp[i][j]
            near_len[i, j, :]                = near_len_local[i][j]
            for k, s in enumerate(b["near_stats_list"][j]):
                near_stat_pad[i, j, k] = s

    # ========== δxy / δcs ==========
    dxy_pad = torch.zeros(B, n_B_max, K, 2)
    dcs_pad = torch.zeros(B, n_B_max, K, 2)
    for i, b in enumerate(batch):
        n_bi = b["delta_xy"].shape[0]
        dxy_pad[i, :n_bi, :] = b["delta_xy"]
        dcs_pad[i, :n_bi, :] = b["delta_cs"]

    # ========== dist_seg ==========
    dist_pad = torch.zeros(B, n_B_max)
    for i, b in enumerate(batch):
        dist_pad[i, : b["dist_seg"].numel()] = b["dist_seg"]

    # ========== speed_A ==========
    speedA_pad = torch.stack([b["speed_A"] for b in batch])  # (B,)

    # ========== B6 ==========
    B6_pad = torch.zeros(B, n_B_max, 6)
    for i, b in enumerate(batch):
        B6_pad[i, : b["B6_list"].shape[0], :] = b["B6_list"]

    # ========== label ==========
    label_pad = torch.stack([b["label"] for b in batch])  # (B,)

    # ========== news_feat（可选） ==========
    news_pad = None
    if batch[0].get("news_feat") is not None:
        d_in    = batch[0]["news_feat"].shape[-1]
        nB_news = max(b["news_feat"].shape[1] for b in batch)
        M       = max(b["news_feat"].shape[2] for b in batch)
        news_pad = torch.zeros(B, nB_news, M, d_in)
        for i, b in enumerate(batch):
            nb, m = b["news_feat"].shape[1:3]
            news_pad[i, :nb, :m] = b["news_feat"]

    # # ===================== DEBUG 输出 =====================
    # print(f"\n[DEBUG collate_fn_eta] batch size B = {B}")
    # print(f"  A_raw_pad.shape    = {A_raw_pad.shape}")
    # print(f"  A_proj_pad.shape   = {A_proj_pad.shape}")
    # print(f"  A_len.shape        = {A_len.shape}")
    # print(f"  A_stat_pad.shape   = {A_stat_pad.shape}\n")
    #
    # print(f"  ship_raw_pad.shape  = {ship_raw_pad.shape}")
    # print(f"  ship_proj_pad.shape = {ship_proj_pad.shape}")
    # print(f"  ship_len.shape      = {ship_len.shape}")
    # print(f"  ship_stat_pad.shape = {ship_stat_pad.shape}\n")
    #
    # print(f"  near_raw_pad.shape  = {near_raw_pad.shape}")
    # print(f"  near_proj_pad.shape = {near_proj_pad.shape}")
    # print(f"  near_len.shape      = {near_len.shape}")
    # print(f"  near_stat_pad.shape = {near_stat_pad.shape}")
    # print(f"  dxy_pad.shape       = {dxy_pad.shape}")
    # print(f"  dcs_pad.shape       = {dcs_pad.shape}\n")
    #
    # print(f"  dist_pad.shape      = {dist_pad.shape}")
    # print(f"  speedA_pad.shape    = {speedA_pad.shape}")
    # print(f"  B6_pad.shape        = {B6_pad.shape}")
    # print(f"  label_pad.shape     = {label_pad.shape}")
    # if news_pad is not None:
    #     print(f"  news_pad.shape      = {news_pad.shape}")
    # print("=====================================================\n")
    del batch
    torch.cuda.empty_cache()
    return (
        A_raw_pad,      # (B, T_A_max, 7)
        A_proj_pad,     # (B, n_B_max, T_A_max, 7)
        A_len,          # (B,)
        A_stat_pad,     # (B, 16)

        ship_raw_pad,   # (B, H, T_ship_pad, 7)
        ship_proj_pad,  # (B, n_B_max, H, T_ship_pad, 7)
        ship_len,       # (B, H)
        ship_stat_pad,  # (B, H, 16)

        near_raw_pad,   # (B, n_B_max, K, T_near_pad, 7)
        near_proj_pad,  # (B, n_B_max, K, T_near_pad, 7)
        near_len,       # (B, n_B_max, K)
        near_stat_pad,  # (B, n_B_max, K, 16)

        dxy_pad,        # (B, n_B_max, K, 2)
        dcs_pad,        # (B, n_B_max, K, 2)

        dist_pad,       # (B, n_B_max)
        speedA_pad,     # (B,)
        B6_pad,         # (B, n_B_max, 6)
        label_pad,      # (B,)
        news_pad        # (B, nB_news, M, d_in) or None
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



@torch.no_grad()
def eval_eta(mdl,
             Aemb,
             shipemb,
             nearemb,
             val_dl,
             device: torch.device,
             criterion,
             use_amp: bool = True,
             news_enc=None) -> float:
    """Run one full validation epoch and return the mean loss/metric.

    Parameters
    ----------
    news_enc : callable or None
        If provided, used to embed the padded news tensor.
    """
    mdl.eval()
    Aemb.eval()
    shipemb.eval()
    nearemb.eval()

    tot_loss = 0.0
    tot_abs = 0.0
    tot_sq = 0.0
    n_seen = 0

    with torch.no_grad():
        for step, batch in enumerate(val_dl):
            if 0 < 10 <= step:
                break
            print(f'验证第{step}开始')
            # ---------------- unpack & to device ----------------
            (A_raw, A_proj, A_len, A_stat,
             ship_raw, ship_proj, ship_len, ship_stat,
             near_raw, near_proj, near_len, near_stat,
             dxy, dcs, dist_seg, speed_A,
             B6, label,
             news_feat) = batch

            A_raw  = A_raw.to(device)
            A_proj = A_proj.to(device)
            A_len  = A_len.to(device)
            A_stat = A_stat.to(device)

            ship_raw  = ship_raw.to(device)
            ship_proj = ship_proj.to(device)
            ship_len  = ship_len.to(device)
            ship_stat = ship_stat.to(device)

            near_raw  = near_raw.to(device)
            near_proj = near_proj.to(device)
            near_len  = near_len.to(device)
            near_stat = near_stat.to(device)

            dxy      = dxy.to(device)
            dcs      = dcs.to(device)
            dist_seg = dist_seg.to(device)
            speed_A  = speed_A.to(device)

            B6    = B6.to(device)
            label = label.to(device)

            if news_feat is not None:
                news_feat = news_feat.to(device)

            # ---------------- forward ----------------
            ctx = autocast(device_type='cuda', enabled=use_amp) if _AMP_NEW else autocast(enabled=use_amp)
            with ctx:
                # ---- A embedding （必须先 unsqueeze 成 (B,1,T_A,7) / (B,nB,1,T_A,7) / (B,1) / (B,1,16)） ----
                A_seq_raw  = A_raw.unsqueeze(1)    # (B, 1, T_A, 7)
                A_seq_proj = A_proj.unsqueeze(2)   # (B, nB, 1, T_A, 7)
                A_lengths  = A_len.unsqueeze(1)    # (B, 1)
                A_stat_exp = A_stat.unsqueeze(1)   # (B, 1, 16)

                A_emb = Aemb(
                    A_seq_raw,    # (B, 1, T_A, 7)
                    A_seq_proj,   # (B, nB, 1, T_A, 7)
                    A_lengths,    # (B, 1)
                    A_stat_exp    # (B, 1, 16)
                )  # → (B, nB, 1, 128)
                # ---- ship embedding （保持原调用格式） ----
                ship_emb = shipemb(
                    ship_raw,    # (B, H, T_ship, 7)
                    ship_proj,   # (B, nB, H, T_ship, 7)
                    ship_len,    # (B, H)
                    ship_stat    # (B, H, 16)
                )  # → (B, nB, H, 128)
                # ---- near embedding （保持原调用格式） ----
                near_emb = nearemb(
                    near_raw,    # (B, nB, K, T_near, 7)
                    near_proj,   # (B, nB, K, T_near, 7)
                    near_len,    # (B, nB, K)
                    near_stat    # (B, nB, K, 16)
                )  # → (B, nB, K, 128)

                # ---- news embedding (可选) ----
                if (news_feat is not None) and (news_enc is not None):
                    news_emb = news_enc(news_feat)
                else:
                    nB = near_emb.size(1)
                    news_emb = torch.zeros(A_emb.size(0), nB, mdl.d_news, device=device)

                # ---- predictor ----
                pred = mdl(
                    B6,
                    A_emb,
                    near_emb,
                    dxy,
                    dcs,
                    ship_emb,
                    dist_seg,
                    speed_A,
                    news_emb
                ).squeeze(-1)  # (B,)
                loss = criterion(pred, label)

            bsz = label.size(0)
            tot_loss += loss.item() * bsz
            tot_abs += torch.sum(torch.abs(pred - label)).item()
            tot_sq += torch.sum((pred - label) ** 2).item()
            n_seen += bsz

    mdl.train()
    Aemb.train()
    shipemb.train()
    nearemb.train()

    if n_seen == 0:
        warnings.warn("Validation loader returned no samples; metrics are set to 0")
        return 0.0, 0.0, 0.0

    mare = tot_loss / n_seen
    mae = tot_abs / n_seen
    rmse = math.sqrt(tot_sq / n_seen)

    return mare, mae, rmse




def compute_hermite_distances(lats, lons, courses, R=6371000.0):
    """
    输入：
      lats:    (n,) 纬度序列（°）
      lons:    (n,) 经度序列（°）
      courses: (n,) 航向序列（°，0°=正北，顺时针）
      R:       地球半径，默认 6 371 000 m

    返回：
      dists_nm: (n-1,) 每对相邻点之间的弧长（海里）。若两点投影后重合，则该段距离 = 0。
    """
    # 1) 投影到局部平面 (米)
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    lat0 = lat_rad[0]
    X = R * (lon_rad - lon_rad[0]) * np.cos(lat0)
    Y = R * (lat_rad - lat0)

    n = len(X)
    # 2) 计算切向量（dx/dt, dy/dt）: 方向由航向给出
    cr_rad = np.deg2rad(courses)
    dX = np.sin(cr_rad)  # 对应 X 方向的切向量分量
    dY = np.cos(cr_rad)  # 对应 Y 方向的切向量分量

    # 3) 计算累积参数 t：任意一对邻点 (i, i+1)，它们的 t 差就是两点在平面上的欧氏距离
    t = np.zeros(n)
    planar_steps = np.hypot(np.diff(X), np.diff(Y))
    t[1:] = np.cumsum(planar_steps)

    # 4) 对每对相邻点分别计算：
    dists = np.zeros(n - 1, dtype=float)  # 单位：米
    for i in range(n - 1):
        if planar_steps[i] == 0:
            dists[i] = 0.0
            continue

        t0, t1 = t[i], t[i + 1]
        x0, x1 = X[i], X[i + 1]
        y0, y1 = Y[i], Y[i + 1]
        dx0, dx1 = dX[i], dX[i + 1]
        dy0, dy1 = dY[i], dY[i + 1]

        sx = CubicHermiteSpline([t0, t1], [x0, x1], [dx0, dx1])
        sy = CubicHermiteSpline([t0, t1], [y0, y1], [dy0, dy1])

        integrand = lambda u: np.hypot(sx(u, 1), sy(u, 1))
        dist_seg, _ = quad(integrand, t0, t1)
        dists[i] = dist_seg

    # 米 转 海里（1 海里 = 1852 米）
    dists_nm = dists / 1852.0
    return dists_nm



def get_node_related_news_tensor(nodes, UN_PRED_NEWS, PRED_NEWS, max_news_num=10):
    """
    input:
      nodes: 节点列表, 每个节点是一个字典, 包含 timestamp, longitude, latitude 字段
      UN_PRED_NEWS: pd.read_csv(unpredictable.csv) 的测试集/训练集
      PRED_NEWS: pd.read_csv(predictable.csv) 的测试集/训练集
      max_num: 每个节点最多选择的新闻数量
    return:
      torch.Tensor shape (idx_of_nodes, idx_of_news, 16)
      2(event_time, delta_t) + 6(scores) +
      4(north, south, east, west) +
      4(north-lat, south-lat, east-lon, west-lon)
    """
    cols = ["event_time", "north", "south", "east", "west"] + [f"score_{i}" for i in range(6)]
    up_arr  = UN_PRED_NEWS[cols].values
    pr_arr  = PRED_NEWS [cols].values

    up_times = up_arr[:,0]
    pr_times = pr_arr[:,0]

    num_nodes = len(nodes)
    out = torch.zeros((num_nodes, max_news_num, 16), dtype=torch.float32)

    for idx, node in enumerate(nodes):
        t, lat, lon = node["timestamp"], node["latitude"], node["longitude"]
        if isinstance(t, datetime):
            t = t.timestamp() 
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
            hits = hits[:max_news_num]
            times    = hits[:, 0:1]
            delta_t  = times - t
            north    = hits[:, 1:2]
            south    = hits[:, 2:3]
            east     = hits[:, 3:4]
            west     = hits[:, 4:5]
            scores   = hits[:, 5:11]

            # 差值特征：新闻范围与节点位置的差
            diff_feat = np.hstack((north - lat,
                                   south - lat,
                                   east - lon,
                                   west - lon))

            tensor   = torch.tensor(
                np.hstack((times, delta_t, scores, north, south, east, west, diff_feat)),
                dtype=torch.float32
            )

            out[idx, :tensor.shape[0], :] = tensor

    return out