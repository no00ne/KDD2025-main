"""
utils.py  ────────────────────────────────────────────────────────────
项目共用工具：
  • 随机种子
  • Checkpoint I/O
  • 计时 / 日志
  • 时间‑空间编码
  • Path / Stat 张量构造
  • collate_fn  — 支持跨样本时间维补齐
"""
import os, json, math, random, logging, time
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

import torch

# ========== AMP 兼容导入 ==========
try:    # torch >= 2.3
    from torch.amp import autocast, GradScaler
    _AMP_NEW = True
except ImportError:             # torch 1.10 ~ 2.2
    from torch.cuda.amp import autocast, GradScaler
    _AMP_NEW = False

# -------------------- 随机 --------------------
def set_seed(seed=2025):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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
    logging.info(f"{name} ▶ end   (%.3f s)", time.perf_counter()-t0)

# -------------------- 编码工具 --------------------
def encode_time(ts):
    if not isinstance(ts, str):
        ts = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
    dt = datetime.fromisoformat(ts)
    h = dt.hour + dt.minute/60 + dt.second/3600
    rad = 2*math.pi*h/24
    return math.sin(rad), math.cos(rad)

def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    φ1,φ2,Δφ,Δλ = map(math.radians, (lat1,lat2,lat2-lat1,lon2-lon1))
    a = math.sin(Δφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(Δλ/2)**2
    return 2*R*math.asin(math.sqrt(a))

def latlon_to_local(lat0, lon0, head, lat, lon, R_km=20.0):
    dx = haversine_km(lat0, lon0, lat0, lon) * math.copysign(1, lon-lon0)
    dy = haversine_km(lat0, lon0, lat, lon0) * math.copysign(1, lat-lat0)
    θ  = math.radians(head)
    x_ = dx*math.cos(-θ)-dy*math.sin(-θ)
    y_ = dx*math.sin(-θ)+dy*math.cos(-θ)
    return x_/R_km, y_/R_km

# -------------------- 特征构造 --------------------
def build_seq_tensor(nodes:list, ref):
    lat0,lon0,head0 = ref["latitude"], ref["longitude"], ref.get("course",0.0)
    feats=[]
    for n in nodes:
        spd = (n.get("speed") or 0.0)*1.852
        sin_c,cos_c = math.sin(math.radians(n.get("course",0.0))), math.cos(math.radians(n.get("course",0.0)))
        dx,dy = latlon_to_local(lat0,lon0,head0,n["latitude"],n["longitude"])
        sin_h,cos_h = encode_time(n.get("timestamp",""))
        feats.append([spd,sin_c,cos_c,dx,dy,sin_h,cos_h])
    return torch.tensor(feats,dtype=torch.float)

def build_stat_tensor(v:dict):
    return torch.tensor([
        v.get("width",0.0), v.get("length",0.0), v.get("path_len",0.0),
        v.get("start_lat",0.0), v.get("start_lon",0.0),
        v.get("end_lat",0.0),   v.get("end_lon",0.0)
    ], dtype=torch.float)

# -------------------- padding --------------------
def _pad_inner(seqs:list[torch.Tensor])->torch.Tensor:
    """
    list[(T_i,7)] → (N, T_max, 7).  若全部空则返回 (N,1,7) 全零
    """
    if len(seqs)==0: return torch.zeros(0,1,7)
    T = max(s.shape[0] for s in seqs) or 1
    out=torch.zeros(len(seqs),T,7)
    for i,s in enumerate(seqs): out[i,:s.shape[0]] = s
    return out

def _pad_batch_time(tensors:list[torch.Tensor])->torch.Tensor:
    """
    list[(K,T_i,7)] → (B,K,T_max,7)
    """
    if len(tensors)==0: return torch.zeros(0)
    K = tensors[0].shape[0]
    T = max(t.shape[1] for t in tensors) or 1
    out = torch.zeros(len(tensors),K,T,7)
    for i,t in enumerate(tensors): out[i,:,:t.shape[1]] = t
    return out

# -------------------- collate_fn --------------------
def collate_fn(batch):
    # -------- A --------
    A_seq  = _pad_inner([b["A_seq"] for b in batch])        # (B,T_A,7)
    A_len  = torch.tensor([b["A_seq"].shape[0] for b in batch])
    A_stat = torch.stack([b["A_stat"] for b in batch])      # (B,7)

    # -------- Near ships --------
    near_seq  = _pad_batch_time([_pad_inner(b["near_seqs"]) for b in batch])
    near_len  = torch.tensor([[s.shape[0] for s in b["near_seqs"]] for b in batch])
    near_stat = torch.stack([torch.stack(b["near_stats"]) for b in batch])
    dxy       = torch.stack([b["delta_xy"] for b in batch])
    dcs       = torch.stack([b["delta_cs"] for b in batch])

    # -------- History --------
    ship_seq  = _pad_batch_time([_pad_inner(b["ship_seqs"]) for b in batch])
    ship_len  = torch.tensor([[s.shape[0] for s in b["ship_seqs"]] for b in batch])
    ship_stat = torch.stack([torch.stack(b["ship_stats"]) for b in batch])

    # -------- Point B features & label --------
    B6    = torch.stack([b["B_feat6"] for b in batch])
    label = torch.stack([b["label"]   for b in batch])

    return (A_seq,A_len,A_stat,
            near_seq,near_len,near_stat,dxy,dcs,
            ship_seq,ship_len,ship_stat,
            B6,label)

# -------------------- evaluate --------------------
@torch.no_grad()
def evaluate(model,embedder,loader,device,criterion,amp=True):
    model.eval(); embedder.eval()
    tot, cnt = 0.0,0
    for batch in loader:
        batch=[x.to(device) for x in batch]
        (A_seq,A_len,A_stat,
         near_seq,near_len,near_stat,dxy,dcs,
         ship_seq,ship_len,ship_stat,
         B6,label)=batch
        B,K,Tn,_ = near_seq.shape
        H,Ts     = ship_seq.shape[1:3]
        ctx = autocast(device_type='cuda',enabled=amp) if _AMP_NEW else autocast(enabled=amp)
        with ctx:
            A_emb   = embedder(A_seq,A_len,A_stat)
            near_emb= embedder(
                near_seq.reshape(B*K,Tn,7),
                near_len.reshape(B*K),
                near_stat.reshape(B*K,7)
            ).view(B,K,-1)
            ship_emb= embedder(
                ship_seq.reshape(B*H,Ts,7),
                ship_len.reshape(B*H),
                ship_stat.reshape(B*H,7)
            ).view(B,H,-1)
            pred = model(B6,A_emb,near_emb,dxy,dcs,ship_emb).squeeze(-1)
            loss = criterion(pred,label)
        tot+=loss.item()*B; cnt+=B
    model.train(); embedder.train()
    return tot/cnt
