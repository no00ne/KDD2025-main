import math
import torch
import torch.nn as nn
from eta_speed_model import NearAggregator

class NewsEncoder(nn.Module):
    """
    简单示例：将每条新闻的 BERT-CLS(768) → 128；
    对同一个 B 参考点的多条新闻做 mean-pooling。
    """
    def __init__(self, d_in: int = 768, d_out: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.GELU(),
            nn.Linear(d_out, d_out)
        )

    def forward(self, news_feat: torch.Tensor):    # (B,nB,M,d_in)
        # 先对 M 篇取 mean => (B,nB,d_in)
        news_mean = news_feat.mean(dim=-2)
        # 再线性映射 => (B,nB,d_out)
        return self.proj(news_mean)

class ETAPredictorNet(nn.Module):
    """
    ETA predictor network: directly predicts ETA (remaining hours) from given context.
    """
    def __init__(self, d_emb=128):
        super().__init__()
        # Use a similar structure as SpeedPredictor but output ETA
        self.near_aggr = NearAggregator(d_emb=d_emb, d_q=64)
        self.B_proj = nn.Linear(6, 64)
        # Fuse vector size: A_emb (128) + nearP (128) + shipP (128) + B_proj (64) = 448
        self.mlp = nn.Sequential(
            nn.Linear(d_emb * 3 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, B6, A_emb, near_emb, delta_xy, delta_cs, ship_emb):
        """
        Inputs:
          B6       : (B, 6) feature of current point (time + heading)
          A_emb    : (B, 128) embedding of trajectory up to A
          near_emb : (B, K, 128) embeddings of K nearby voyages
          delta_xy : (B, K, 2) relative coordinates of neighbor start
          delta_cs : (B, K, 2) cos/sin of course difference
          ship_emb : (B, H, 128) embeddings of H historical voyages
        Output:
          ETA prediction (B, 1)
        """
        # Project B features
        Bq = self.B_proj(B6)  # (B,64)
        # Aggregate nearby voyages
        nearP = self.near_aggr(near_emb, delta_xy, delta_cs, Bq)  # (B,128)
        # Aggregate ship history by mean
        shipP = ship_emb.mean(1)  # (B,128)
        # Fuse features
        fuse = torch.cat([A_emb, nearP, shipP, Bq], dim=-1)  # (B, 128*3 + 64)
        # Predict ETA (hours)
        eta = self.mlp(fuse)  # (B,1)
        return eta
