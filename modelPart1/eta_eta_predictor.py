import torch
from eta_speed_model import MultiHeadNearAggregator

import torch.nn as nn


class ETAPredictorNet(nn.Module):
    def __init__(self, use_news: bool = False, d_emb: int = 128, d_news: int = 128, heads: int = 4):
        super().__init__()
        self.use_news = use_news
        self.d_news = d_news
        self.near_aggr = MultiHeadNearAggregator(d_emb, 64, heads)
        self.B_proj = nn.Linear(6, 64)

        # 用于把 Bq_f 投影到和 A/near/ship 同维度
        self.q_proj = nn.Linear(64, d_emb)

        # MultiHeadAttention：key/value 都是三类特征拼接后的序列 (长度=3)
        self.attn = nn.MultiheadAttention(embed_dim=d_emb, num_heads=heads, batch_first=True)

        fuse_in_dim = d_emb  # attention 输出就是 d_emb
        if use_news:
            fuse_in_dim += d_news

        # attention 之后再接一个小型 MLP 或直接线性映射
        self.speed_head = nn.Sequential(nn.Linear(fuse_in_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, B6, A_emb,  # (B, nB, 1, 128)
                near_emb,  # (B, nB, K, 128)
                delta_xy,  # (B, nB, K, 2)
                delta_cs,  # (B, nB, K, 2)
                ship_emb,  # (B, nB, H, 128)
                dist_seg,  # (B, nB)
                speed_A,  # (B,)
                news_emb=None  # (B, nB, 128)
                ):
        B, nB, K = near_emb.shape[:3]
        H = ship_emb.size(2)

        # flatten batch & nB
        B6_f = B6.view(B * nB, -1)  # (B*nB, 6)
        Bq = self.B_proj(B6_f)   # (B*nB, 64)
        Bq_f = self.q_proj(Bq)   # (B*nB, 128)

        # 1) 计算 nearP_f
        nearP_f = self.near_aggr(
            near_emb.view(B * nB, K, -1),
            delta_xy.view(B * nB, K, -1),
            delta_cs.view(B * nB, K, -1),
            Bq
        )  # (B*nB, 128)

        # 2) 计算 shipP_f：对 H 维度做 attention，而不是简单均值
        #    先把 ship_emb_flat 视作序列 (seq_len=H)
        ship_flat = ship_emb.view(B * nB, H, -1)  # (B*nB, H, 128)
        #    用 Bq_f 作为 query，对 ship_flat 做一次点乘注意力
        #    reshape成 (B*nB, 1, 128) 才能作为 query
        q = Bq_f.unsqueeze(1)  # (B*nB, 1, 128)
        shp_attn, _ = self.attn(q,  # query: (B*nB, 1, 128)
                                ship_flat,  # key:   (B*nB, H, 128)
                                ship_flat)  # value: (B*nB, H, 128)
        shipP_f = shp_attn.squeeze(1)  # (B*nB, 128)

        # 3) 取出 A_emb 并 flatten
        A_squeezed = A_emb.squeeze(2)  # (B, nB, 128)
        A_f = A_squeezed.view(B * nB, -1)  # (B*nB, 128)

        # 4) 拼接三类特征，再做一次注意力融合：
        #    把 A_f、nearP_f、shipP_f 拼成一个长度为3的"序列"
        feats = torch.stack([A_f, nearP_f, shipP_f], dim=1)  # (B*nB, 3, 128)
        #    把 Bq_f (已经是 128 维) 作为最终 query
        q2 = Bq_f.unsqueeze(1)  # (B*nB, 1, 128)
        attn_out, _ = self.attn(q2,  # query: (B*nB,1,128)
                                feats,  # key:   (B*nB,3,128)
                                feats)  # value: (B*nB,3,128)
        fuse_feat = attn_out.squeeze(1)  # (B*nB, 128)

        # 5) 如果有 news_emb，再拼上去
        if self.use_news and (news_emb is not None):
            news_f = news_emb.view(B * nB, -1)  # (B*nB,128)
            fuse_feat = torch.cat([fuse_feat, news_f], dim=-1)  # (B*nB, 256)

        # 6) 过 MLP 或线性映射得到速度预测
        speed_pred = self.speed_head(fuse_feat).view(B, nB)  # (B, nB)

        # 下面保持原逻辑：根据 speed_A + speed_pred 做物理累加算 ETA
        speed_pad = torch.cat([speed_A.unsqueeze(1), speed_pred], dim=1)  # (B, nB+1)
        print('speed_pad')
        print(speed_pad[:5, :5])
        seg_speed = 0.5 * (speed_pad[:, :-1] + speed_pad[:, 1:])  # (B, nB)
        print('seg_speed')
        print(dist_seg[:5, :5])

        eps = 1e-3
        seg_time = dist_seg / (seg_speed + eps)  # hours
        eta = seg_time.sum(dim=1)  # (B,)

        return eta


