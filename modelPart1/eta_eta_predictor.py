import math
import torch
import torch.nn as nn
from eta_speed_model import NearAggregator



class ETAPredictorNet(nn.Module):
    """
    输入（括号内形状）：
      • B6           : (B , nB , 6)
      • A_emb        : (B , 128)                  —— 对应 A_raw
      • near_emb     : (B , nB , K , 128)
      • delta_xy     : (B , nB , K , 2)
      • delta_cs     : (B , nB , K , 2)
      • ship_emb     : (B , nB , H , 128)        —— broadcast-cache 后的
      • dist_seg     : (B , nB)                —— 相邻 B-node 的弧长 km
      • speed_A      : (B ,)                     —— A 点 km/h
      • news_emb (opt): (B , nB , 128)           —— 若 use_news=True
    输出：
      • eta_pred     : (B ,)   —— 预测的剩余小时数
    """
    def __init__(self,
                 use_news: bool = False,
                 d_emb: int = 128,
                 d_news: int = 128):
        super().__init__()
        self.use_news = use_news
        self.near_aggr = NearAggregator(d_emb, 64)             # 见原实现
        self.B_proj    = nn.Linear(6, 64)

        fuse_in_dim = d_emb * 3 + 64               # A + near + ship + Bq
        if use_news:
            fuse_in_dim += d_news

        # 输出 **速度(km / h)**，每个 B-node 一条
        self.speed_head = nn.Sequential(
            nn.Linear(fuse_in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),          nn.ReLU(),
            nn.Linear(64, 1)                           # → scalar speed
        )

    # -----------------------------------------------------------
    def forward(self,
                B6, A_emb,
                near_emb, delta_xy, delta_cs,
                ship_emb,                       # (B,nB,H,128)
                dist_seg,                       # (B,nB-1)
                speed_A,
                news_emb: torch.Tensor | None = None        # (B,nB,128)
                ):
        B, nB, K = near_emb.shape[:3]
        H = ship_emb.size(2)

        # -------- reshape 方便并行计算 --------
        # flatten batch & nB 维： (B*nB , K , …)
        near_emb_f = near_emb.view(B * nB, K, -1)
        dx_f       = delta_xy.view(B * nB, K, -1)
        dc_f       = delta_cs.view(B * nB, K, -1)
        # 同理把 B6、ship 拍平成 (B*nB , …)
        B6_f   = B6.view(B * nB, -1)                      # (BnB,6)
        ship_f = ship_emb.view(B * nB, H, -1).mean(1)     # (BnB,128) ← 平均聚合
        A_f    = A_emb.repeat_interleave(nB, dim=0)       # (BnB,128)

        # -------------- 对每个 B-node 做邻船注意力聚合 --------------
        Bq_f    = self.B_proj(B6_f)                       # (BnB,64)
        nearP_f = self.near_aggr(near_emb_f, dx_f, dc_f, Bq_f)   # (BnB,128)

        # 拼接所有特征
        fuse = torch.cat([A_f, nearP_f, ship_f, Bq_f], dim=-1)    # (BnB, 128*3+64)
        if self.use_news and news_emb is not None:
            news_f = news_emb.view(B * nB, -1)                    # (BnB,128)
            fuse = torch.cat([fuse, news_f], dim=-1)

        speed_pred = self.speed_head(fuse).view(B, nB)            # (B,nB)

        # -----------   ETA 物理累计   -----------
        #   seg_speed = (v_i + v_{i+1}) / 2
        speed_pad = torch.cat([speed_A.unsqueeze(1), speed_pred], dim=1)  # (B,nB+1)
        seg_speed = 0.5 * (speed_pad[:, :-1] + speed_pad[:, 1:])  # (B,nB)
        #   time_i   = dist_i / (seg_speed_i + ε)
        eps = 1e-3
        seg_time = dist_seg / (seg_speed + eps)                     # hours
        eta = seg_time.sum(dim=1)                                   # (B,)

        return eta
