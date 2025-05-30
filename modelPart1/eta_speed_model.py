"""
eta_speed_model.py
─────────────────────────────────────────────────────────
包含：NodeAEmbedder, StaticEmbedder, GroupEmbedder,
     NearAggregator, SpeedPredictor
"""
import math

import torch
import torch.nn as nn


class NewsEmbedder(nn.Module):
    """
    预留：新闻文本 / 多模态新闻特征的嵌入器。
    目前仅占位，forward 直接抛出 NotImplementedError。
    你可以在此处接入任意 BERT / CLIP / LLM + Pooling 等。
    """

    def __init__(self, d_in: int = 768, d_out: int = 128):
        super().__init__()
        # >>> 在这里实现你的投射 / pooling 层 <<<
        raise NotImplementedError("请在 NewsEmbedder 内部实现具体的嵌入逻辑")

    def forward(self, news_feat: torch.Tensor) -> torch.Tensor:
        """
        news_feat : (B, nB, M, d_in)
          B   —— batch
          nB  —— 每个参考点 B 对应的新闻条数（可变，已在 collate 中 pad）
          M   —— 单条新闻的 token / patch / frame 维度（可变）
        返回 : (B, nB, d_out)
        """
        raise NotImplementedError


class FuseBlock(nn.Module):
    """ raw-128 与 proj-128 → 128  """

    def __init__(self, d=128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d))

    def forward(self, x_raw, x_proj):  # (...,d) × 2
        return self.fuse(torch.cat([x_raw, x_proj], dim=-1))


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
     轨迹序列 → 向量 (d_model)。
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


class GroupEmbedder(nn.Module):
    """
    (seq_raw , seq_proj , len , stat) → (N , 128)
    """

    def __init__(self,
                 d_seq_in: int = 7,
                 d_stat_in: int = 16):
        super().__init__()
        d_model = 64
        self.enc = NodeAEmbedder(d_in=d_seq_in, d_model=d_model)
        self.fuse = FuseBlock(d=d_model)
        self.stat = StaticEmbedder(d_in=d_stat_in, d_emb=d_model)

    def forward(self,
                seq_raw: torch.Tensor,  # (N,T,7)
                seq_proj: torch.Tensor,  # (N,T,7)
                lengths: torch.Tensor,  # (N,)
                stat: torch.Tensor):  # (N,16)
        h_raw = self.enc(seq_raw, lengths)
        h_proj = self.enc(seq_proj, lengths)
        h_traj = self.fuse(h_raw, h_proj)  # (N,64)
        h_stat = self.stat(stat)  # (N,64)
        return torch.cat([h_traj, h_stat], dim=-1)  # (N,128)


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
