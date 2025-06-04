"""
eta_speed_model.py
─────────────────────────────────────────────────────────
包含：NodeAEmbedder, StaticEmbedder, GroupEmbedder,
     NearAggregator, SpeedPredictor
"""
import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


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


class LearnablePosEncoding(nn.Module):
    """
    学习式位置编码，比固定 Sine-Cos 更灵活。
    max_len 取一个够大的上界（> 数据中最长序列），会自动截断或补零。
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        # 原先的 pe 长度为 max_len；如果实际序列更长，就补零
        self.pe = nn.Parameter(torch.zeros(max_len, d_model))  # (max_len, d_model)
        nn.init.normal_(self.pe, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, valid_len: torch.LongTensor):
        """
        x         : (B, T, d_model) —— 线性映射后的特征
        valid_len : (B,)            —— 每条有效步长
        返回同 shape x，已加位置编码（对超出有效长度的 padding 步，不加）。
        """
        B, T, _ = x.shape
        max_pe, d_model = self.pe.shape

        if T <= max_pe:
            # 序列长度在 pe 范围内，直接切片
            pe = self.pe[:T].unsqueeze(0)     # (1, T, d_model)
        else:
            # 序列比 max_len 还长，需要把 pe[:max_pe] 和后面补零拼接成 (T, d_model)
            # 先把已有的 pe[:max_pe] 拷贝一次，再补 (T - max_pe) 条全零
            extra = torch.zeros((T - max_pe, d_model), device=x.device, dtype=x.dtype)
            pe_full = torch.cat([self.pe, extra], dim=0)  # (max_pe + (T-max_pe) = T, d_model)
            pe = pe_full.unsqueeze(0)  # (1, T, d_model)

        # 生成 mask：padding 步不加位置编码
        idx = torch.arange(T, device=x.device).view(1, -1)  # (1, T)
        mask = (idx < valid_len.view(-1, 1)).float()       # (B, T)

        return x + pe * mask.unsqueeze(-1)  # (B, T, d_model)

# -------------------- 序列编码器（Transformer） --------------------
class NodeAEmbedder(nn.Module):
    def __init__(self,
                 d_in: int = 7,
                 d_model: int = 64,
                 n_head: int = 8,
                 n_layer: int = 2,
                 ff_mult: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 4096):
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
        seq     : (N, T_max, 7)          —— 右侧 0-padding
        lengths : (N,)                    —— 每条有效步长
        返回   : (N, d_model)
        """
        x = self.proj(seq)                     # (N, T, d_model)
        x = self.pos(x, lengths)               # 加位置编码
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        z = self.encoder(x, src_key_padding_mask=mask)  # (N, T, d_model)
        lens = lengths.clamp(min=1).view(-1, 1)         # 避免除 0
        out = (z * (~mask).unsqueeze(-1)).sum(1) / lens # (N, d_model)
        return out

# -------------------- 静态特征编码（MLP） --------------------
class StaticEmbedder(nn.Module):
    def __init__(self, d_in: int = 16, d_emb: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb)
        )

    def forward(self, x: torch.Tensor):
        return self.mlp(x)  # (N, d_emb)

# -------------------- FuseBlock --------------------
class FuseBlock(nn.Module):
    """ raw-64 与 proj-64 → 64 """
    def __init__(self, d: int = 64):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(d * 2, d * 2),
            nn.GELU(),
            nn.Linear(d * 2, d)
        )

    def forward(self, x_raw: torch.Tensor, x_proj: torch.Tensor):
        """
        x_raw  : (..., d)
        x_proj : (..., d)
        返回   : (..., d)
        """
        return self.fuse(torch.cat([x_raw, x_proj], dim=-1))

class NodeAEmbedderGRU(nn.Module):
    """
    用 GRU 替代 Transformer。输入 (N, T, d_in) 和 lengths，输出 (N, d_model)。
    """
    def __init__(self,
                 d_in: int = 7,
                 d_model: int = 64,
                 n_layer: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layer,
            batch_first=True,
            dropout=dropout if n_layer > 1 else 0.0,
            bidirectional=False,
        )
        self.d_model = d_model

    def forward(self, seq: torch.Tensor, lengths: torch.Tensor):
        """
        seq     : (N, T, d_in)      —— 右侧 0-padding
        lengths : (N,)              —— 每条有效步长
        返回   : (N, d_model)
        """
        x = self.proj(seq)                     # (N, T, d_model)
        # 按 lengths 排序，pack，再 run GRU
        lengths_clamped = lengths.clamp(min=1)
        sorted_len, idx_sort = torch.sort(lengths_clamped, descending=True)
        idx_unsort = torch.argsort(idx_sort)
        x_sorted = x[idx_sort]
        packed = pack_padded_sequence(x_sorted, sorted_len.cpu(), batch_first=True)
        packed_out, _ = self.gru(packed)       # 输出 PackedSequence
        out_padded, _ = pad_packed_sequence(packed_out, batch_first=True)  # (N, T, d_model)
        out = out_padded[idx_unsort]           # 恢复原顺序

        # mask & 平均池化
        N, T, D = out.shape
        idx = torch.arange(T, device=out.device)[None, :]
        mask = (idx < lengths.view(-1, 1))     # (N, T)
        sum_h = (out * mask.unsqueeze(-1)).sum(1)            # (N, d_model)
        avg_h = sum_h / lengths_clamped.view(-1, 1)          # (N, d_model)
        return avg_h
# ====================== 修改后的 GroupEmbedder ======================
class GroupEmbedder(nn.Module):
    """
    不同B,nB,m的组相互独立
    支持以下两种输入形式，并统一输出 (B, nB, m, 128)：

    1) seq_raw:  (B, nB, m,  T, 7)
       seq_proj: (B, nB, m,  T, 7)
       lengths:  (B, nB, m)
       stat:     (B, nB, m, 16)

    2) seq_raw:  (B, m,  T, 7)
       seq_proj: (B, nB, m,  T, 7)
       lengths:  (B, m)
       stat:     (B, m, 16)
    """
    def __init__(self,
                 d_seq_in: int = 7,
                 d_stat_in: int = 16,
                 m_news=0,
                 use_news=False,
    ):
        super().__init__()
        self.m_news = m_news
        self.use_news = use_news
        d_model = 64
        self.enc = NodeAEmbedderGRU(d_in=d_seq_in, d_model=d_model)
        self.fuse = FuseBlock(d=d_model)
        self.stat = StaticEmbedder(d_in=d_stat_in, d_emb=d_model)

    def forward(self,
                seq_raw: torch.Tensor,
                seq_proj: torch.Tensor,
                lengths: torch.Tensor,
                stat: torch.Tensor):
        """
        1) 情况1：seq_raw:  (B, nB, m,  T, 7)
                   seq_proj: (B, nB, m,  T, 7)
                   lengths:  (B, nB, m)
                   stat:     (B, nB, m, 16)

        2) 情况2：seq_raw:  (B, m,  T, 7)
                   seq_proj: (B, nB, m,  T, 7)
                   lengths:  (B, m)
                   stat:     (B, m, 16)
        """

        # =====================================================
        # 情况1：B, nB, m, T, D = seq_raw.shape
        # -----------------------------------------------------
        # raw:  (B, nB, m, T, 7)
        # proj: (B, nB, m, T, 7)
        # len:  (B, nB, m)
        # stat: (B, nB, m, 16)
        # =====================================================
        if seq_raw.dim() == 5 and seq_proj.dim() == 5 and lengths.dim() == 3 and stat.dim() == 4:
            B, nB, m, T, D = seq_raw.shape

            # 1) flatten
            raw_flat  = seq_raw.reshape(B * nB * m, T, D)   # (B*nB*m, T, 7)
            proj_flat = seq_proj.reshape(B * nB * m, T, D)  # (B*nB*m, T, 7)
            len_flat  = lengths.reshape(B * nB * m)         # (B*nB*m,)
            stat_flat = stat.reshape(B * nB * m, stat.size(-1))  # (B*nB*m, 16)

            # 2) 对所有 len_flat == 0 的位置，强制把对应的子序列变成 (1, D)，然后长度设为 1
            zero_mask = (len_flat == 0).cpu()
            if zero_mask.any():
                for idx0 in zero_mask.nonzero(as_tuple=False).view(-1).tolist():
                    # 只要原始子序列长度为 0，就用一个 (1, D) 全 0 去替换
                    raw_flat[idx0]  = raw_flat[idx0].new_zeros((1, D))
                    proj_flat[idx0] = proj_flat[idx0].new_zeros((1, D))
                    len_flat[idx0]  = 1

            # 3) encode raw_flat
            h_raw_flat  = self.enc(raw_flat, len_flat)    # (B*nB*m, 64)
            # 4) encode proj_flat（复用已修改后的 len_flat）
            h_proj_flat = self.enc(proj_flat, len_flat)   # (B*nB*m, 64)

            # 5) fuse raw & proj
            h_traj_flat = self.fuse(h_raw_flat, h_proj_flat)  # (B*nB*m, 64)

            # 6) 静态特征编码
            h_stat_flat = self.stat(stat_flat)  # (B*nB*m, 64)

            # 7) 拼接并 reshape
            H_flat = torch.cat([h_traj_flat, h_stat_flat], dim=-1)  # (B*nB*m, 128)
            return H_flat.view(B, nB, m, -1)  # (B, nB, m, 128)

        # =====================================================
        # 情况2：B, nB, m, T, D = seq_proj.shape
        # -----------------------------------------------------
        # raw:  (B, m,   T, 7)
        # proj: (B, nB, m, T, 7)
        # len:  (B, m)
        # stat: (B, m, 16)
        # =====================================================
        if seq_raw.dim() == 4 and seq_proj.dim() == 5 and lengths.dim() == 2 and stat.dim() == 3:
            B, nB, m, T, D = seq_proj.shape

            # 1) flatten raw & lengths & stat
            raw_flat_only = seq_raw.reshape(B * m, T, D)      # (B*m, T, 7)
            len_flat_raw  = lengths.reshape(B * m)            # (B*m,)
            stat_flat_raw = stat.reshape(B * m, stat.size(-1))  # (B*m, 16)

            # —— 对 len_flat_raw == 0 的位置，把 raw_flat_only[idx] 置成 (1, D)，len_flat_raw=1 ——
            zero_mask_raw = (len_flat_raw == 0).cpu()
            if zero_mask_raw.any():
                for idx0 in zero_mask_raw.nonzero(as_tuple=False).view(-1).tolist():
                    # 无论原来的 T 是多少，直接替换成一个 (1, D) 的全零张量
                    raw_flat_only[idx0] = raw_flat_only[idx0].new_zeros((1, D))
                    len_flat_raw[idx0]  = 1

            # 2) encode raw_flat_only
            h_raw_flat_only  = self.enc(raw_flat_only, len_flat_raw)  # (B*m, 64)
            h_stat_flat_only = self.stat(stat_flat_raw)               # (B*m, 64)

            # 3) 扩展到 (B, nB, m)，再 flatten
            h_raw_expand = h_raw_flat_only.view(B, m, -1)                   # (B, m, 64)
            h_raw_expand = h_raw_expand.unsqueeze(1).expand(-1, nB, -1, -1) # (B, nB, m, 64)
            h_raw_flat  = h_raw_expand.reshape(B * nB * m, -1)              # (B*nB*m, 64)

            h_stat_expand = h_stat_flat_only.view(B, m, -1)                  # (B, m, 64)
            h_stat_expand = h_stat_expand.unsqueeze(1).expand(-1, nB, -1, -1) # (B, nB, m, 64)
            h_stat_flat  = h_stat_expand.reshape(B * nB * m, -1)             # (B*nB*m, 64)

            # 4) flatten proj
            proj_flat = seq_proj.reshape(B * nB * m, T, D)  # (B*nB*m, T, 7)

            # 5) 构造 len_flat_proj，并对 len_flat_proj==0 的位置，同样做“(1,D) + length=1”补丁
            len_rep_proj  = lengths.unsqueeze(1).expand(-1, nB, -1)  # (B, nB, m)
            len_flat_proj = len_rep_proj.reshape(B * nB * m)        # (B*nB*m,)

            zero_mask_proj = (len_flat_proj == 0).cpu()
            if zero_mask_proj.any():
                for idx0 in zero_mask_proj.nonzero(as_tuple=False).view(-1).tolist():
                    proj_flat[idx0]    = proj_flat[idx0].new_zeros((1, D))
                    len_flat_proj[idx0] = 1

            # 6) encode proj_flat
            h_proj_flat = self.enc(proj_flat, len_flat_proj)  # (B*nB*m, 64)

            # 7) fuse raw 与 proj
            h_traj_flat = self.fuse(h_raw_flat, h_proj_flat)  # (B*nB*m, 64)

            # 8) 拼接 traj 与 stat
            H_flat = torch.cat([h_traj_flat, h_stat_flat], dim=-1)  # (B*nB*m, 128)

            # —— 释放中间变量显存优化 ——
            del raw_flat_only, stat_flat_raw, h_raw_flat_only, h_stat_flat_only
            del h_raw_expand, h_stat_expand, proj_flat, len_flat_raw, len_flat_proj, h_proj_flat, h_traj_flat
            torch.cuda.empty_cache()

            return H_flat.view(B, nB, m, -1)  # (B, nB, m, 128)

        # 如果都不满足，就报错
        raise ValueError("Unsupported input dimensions for GroupEmbedder")




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
