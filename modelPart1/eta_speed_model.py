"""
eta_speed_model.py
─────────────────────────────────────────────────────────
包含：NodeAEmbedder, StaticEmbedder, GroupEmbedder,
     NearAggregator, SpeedPredictor
"""
import math

import torch
import torch.nn as nn

class FuseBlock(nn.Module):
    """
    将 raw / proj 两个 128-d 向量 → 单一 128-d。
    当前实现：先 concat → 256 → GELU → 128。
    """
    def __init__(self, d_in_each: int = 128, d_out: int = 128):
        super().__init__()
        self.lin = nn.Sequential(
            nn.Linear(d_in_each * 2, d_out * 2),
            nn.GELU(),
            nn.Linear(d_out * 2, d_out)
        )

    def forward(self, x_raw: torch.Tensor, x_proj: torch.Tensor):
        return self.lin(torch.cat([x_raw, x_proj], dim=-1))  # (...,128)

# ------------------------- 位置编码 -------------------------
class LearnablePosEncoding(nn.Module):
    """
    学习式位置编码，比固定 Sine-Cos 更灵活。
    max_len 取一个够大的上界（> 数据中最长序列），会自动截断。
    """

    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))  # (T_max, d_model)
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, valid_len: torch.LongTensor):
        """
        x         : (B, T, d_model) —— 线性映射后的特征
        valid_len : (B,)            —— 每条有效步长
        返回同 shape x，已加位置编码（对超出有效长度的 padding 步，不加）。
        """
        B, T, _ = x.shape
        pe = self.pe[:T].unsqueeze(0)  # (1,T,d_model)
        # padding 步不加位置信息 → mask = (idx < len)
        idx = torch.arange(T, device=x.device).view(1, -1)  # (1,T)
        mask = (idx < valid_len.view(-1, 1)).float()  # (B,T)
        return x + pe * mask.unsqueeze(-1)  # (B,T,d_model)


# ------------------------- NodeAEmbedder -------------------------
class NodeAEmbedder(nn.Module):
    """
    🚢 轨迹序列 → 向量 (d_model)。
    采用 1×Linear + 可学习位置编码 + TransformerEncoder + 有效步长平均池化。
    """

    def __init__(self,
                 d_in: int = 7,
                 d_model: int = 64,
                 n_head: int = 8,
                 n_layer: int = 2,
                 ff_mult: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 1024):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.pos = LearnablePosEncoding(d_model, max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layer)
        self.d_model = d_model

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor):
        """
        参数
        ----
        seq     : (N, T_max, 7)          —— 右侧 0-padding
        lengths : (N,)                   —— 有效步长
        返回
        ----
        out     : (N, d_model)
        """
        x = self.proj(seq)  # (N,T,d_model)
        x = self.pos(x, lengths)  # 加位置编码
        # key_padding_mask: Padding=true → 不参与注意力
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        z = self.encoder(x, src_key_padding_mask=mask)  # (N,T,d_model)
        # 有效步长平均池化
        lens = lengths.clamp(min=1).view(-1, 1)  # 避免除 0
        out = (z * (~mask).unsqueeze(-1)).sum(1) / lens  # (N,d_model)
        return out


# ------------------------- StaticEmbedder -------------------------
class StaticEmbedder(nn.Module):
    """
    16-维航次静态特征 → d_model。 这里用两层 MLP；你可以按需改成更复杂结构。
    """

    def __init__(self, d_in: int = 16, d_emb: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)  # (N,d_emb)


# ------------------------- GroupEmbedder -------------------------
class GroupEmbedder(nn.Module):
    """
    公用编码器：把 **序列轨迹 + 静态特征** 合并成统一的 128-维向量。
    - 对于 A / near / ship 都复用同一份权重，保持语义一致。
    - 输入 **必须** 按顺序给 `(seq, lengths, stat)`。
    """

    def __init__(self,
                 d_seq_in: int = 7,
                 d_stat_in: int = 16,
                 d_model: int = 64):
        super().__init__()
        self.node = NodeAEmbedder(d_in=d_seq_in, d_model=d_model)
        self.stat = StaticEmbedder(d_in=d_stat_in, d_emb=d_model)

    def forward(self,
                seq: torch.Tensor,  # (N,T,7)
                lengths: torch.Tensor,  # (N,)
                stat: torch.Tensor):  # (N,16)
        h_seq = self.node(seq, lengths)  # (N,64)
        h_stat = self.stat(stat)  # (N,64)
        return torch.cat([h_seq, h_stat], dim=-1)  # (N,128)


class NearAggregator(nn.Module):
    """
    方向-距离注意力聚合：给定邻船嵌入 + Δxy + Δcosθ,sinθ 与 B_query (64),
    产生加权池化的近船表示 (128).
    """

    def __init__(self, d_emb=128, d_q=64):
        super().__init__()
        self.key = nn.Linear(d_emb + 4, d_q)
        self.scale = math.sqrt(d_q)

    def forward(self, near_emb, delta_xy, delta_cs, B_query):
        # near_emb: (B,K,128), delta_xy:(B,K,2), delta_cs:(B,K,2), B_query:(B,64)
        Kcat = torch.cat([near_emb, delta_xy, delta_cs], dim=-1)  # (B,K,132)
        Kproj = self.key(Kcat)  # (B,K,64)
        # 点积 / scale -> softmax
        scores = (Kproj * B_query.unsqueeze(1)).sum(-1) / self.scale
        wts = torch.softmax(scores, dim=1)  # (B,K)
        return (wts.unsqueeze(-1) * near_emb).sum(1)  # (B,128)


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
        self.B_proj = nn.Linear(6, 64)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb * 3 + 64, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, B6, A_emb, near_emb, delta_xy, delta_cs, ship_emb):
        # ship_emb: (B,H,128)
        Bq = self.B_proj(B6)  # (B,64)
        nearP = self.near_aggr(near_emb, delta_xy, delta_cs, Bq)  # (B,128)
        shipP = ship_emb.mean(1)  # (B,128)
        fuse = torch.cat([A_emb, nearP, shipP, Bq], dim=-1)  # (B,128*3+64)
        return self.mlp(fuse)  # (B,1)
