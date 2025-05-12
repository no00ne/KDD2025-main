"""
eta_speed_model.py
─────────────────────────────────────────────────────────
包含：NodeAEmbedder, StaticEmbedder, GroupEmbedder,
     NearAggregator, SpeedPredictor
"""
import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class NodeAEmbedder(nn.Module):
    def __init__(self, d_in=7, d_hid=64):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hid, batch_first=True)

    def forward(self, seq, lengths):
        """
        seq      : (B, T_max, 7)   已经右填 0
        lengths  : (B,)  每个样本的真实长度，需降序排
        """
        lengths, perm = lengths.sort(descending=True)
        seq = seq[perm]                          # ① 按长度降序

        packed = pack_padded_sequence(seq, lengths.cpu(), batch_first=True, enforce_sorted=True)
        _, h_n = self.gru(packed)                # ② GRU 只看到有效步长
        h_n = h_n[-1]                            # (B, d_hid)

        # ③ 把 batch 顺序恢复
        _, rev = perm.sort()
        return h_n[rev]                   # (B, d_hid)

class StaticEmbedder(nn.Module):
    def __init__(self, d_in=7, d_emb=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_emb),
            nn.ReLU(),
            nn.Linear(d_emb, d_emb)
        )
    def forward(self, x):
        return self.mlp(x)           # (B, d_emb)

class GroupEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.node = NodeAEmbedder()
        self.stat = StaticEmbedder()

    def forward(self, seq, lengths, stat):
        # seq:(N,T,7)  lengths:(N,)  stat:(N,7)
        return torch.cat([self.node(seq, lengths), self.stat(stat)], dim=-1)
        # 输出 (N,128)


class NearAggregator(nn.Module):
    """
    方向-距离注意力聚合：给定邻船嵌入 + Δxy + Δcosθ,sinθ 与 B_query (64),
    产生加权池化的近船表示 (128).
    """
    def __init__(self, d_emb=128, d_q=64):
        super().__init__()
        self.key = nn.Linear(d_emb+4, d_q)
        self.scale = math.sqrt(d_q)
    def forward(self, near_emb, delta_xy, delta_cs, B_query):
        # near_emb: (B,K,128), delta_xy:(B,K,2), delta_cs:(B,K,2), B_query:(B,64)
        Kcat  = torch.cat([near_emb, delta_xy, delta_cs], dim=-1)  # (B,K,132)
        Kproj = self.key(Kcat)                                     # (B,K,64)
        # 点积 / scale -> softmax
        scores= (Kproj * B_query.unsqueeze(1)).sum(-1) / self.scale
        wts   = torch.softmax(scores, dim=1)                       # (B,K)
        return (wts.unsqueeze(-1) * near_emb).sum(1)               # (B,128)

class SpeedPredictor(nn.Module):
    """
    最终预测下一段速度 (km/h)，输入：
      • B6          : (B,6)
      • A_emb       : (B,128)
      • near_emb    : (B,K,128)
      • delta_xy, delta_cs
      • ship_emb    : (B,H,128)
    """
    def __init__(self, d_emb=128):
        super().__init__()
        self.near_aggr = NearAggregator(d_emb, 64)
        self.B_proj    = nn.Linear(6,64)
        self.mlp       = nn.Sequential(
            nn.Linear(d_emb*3+64,128), nn.ReLU(),
            nn.Linear(128,64),          nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self, B6, A_emb, near_emb, delta_xy, delta_cs, ship_emb):
        # ship_emb: (B,H,128)
        Bq    = self.B_proj(B6)                        # (B,64)
        nearP = self.near_aggr(near_emb, delta_xy, delta_cs, Bq)  # (B,128)
        shipP = ship_emb.mean(1)                       # (B,128)
        fuse  = torch.cat([A_emb, nearP, shipP, Bq], dim=-1)  # (B,128*3+64)
        return self.mlp(fuse)                          # (B,1)
