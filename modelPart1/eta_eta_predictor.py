"""
eta_eta_predictor.py
─────────────────────────────────────────────────────────
面向单次预测的封装：给定一个 JSON 格式的 voyage 及 A_idx，
直接调用数据库 + 上述模型，输出 ETA 时间和剩余小时数。
"""

import json
import psycopg2
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import CubicSpline
import torch

from eta_speed_model      import GroupEmbedder, SpeedPredictor
from utils                import encode_time, latlon_to_local

class ETAPredictor:
    def __init__(self,
                 db_params: dict,
                 model_ckpt: str,
                 K=10, H=5, radius_km=20.0):
        # 1) 加载模型
        self.embedder = GroupEmbedder()
        self.model    = SpeedPredictor()
        ckpt = torch.load(model_ckpt, map_location='cpu')
        self.embedder.load_state_dict(ckpt['embedder'])
        self.model.load_state_dict(ckpt['model'])
        self.embedder.eval(); self.model.eval()

        # 2) DB 连接
        self.conn = psycopg2.connect(**db_params)
        self.cur  = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        self.K = K; self.H = H; self.R = radius_km

    def _fetch_nodes(self, voyage_id):
        self.cur.execute(
            "SELECT * FROM voyage_node WHERE voyage_id=%s ORDER BY timestamp;",
            (voyage_id,))
        return self.cur.fetchall()

    def __call__(self, voyage_json: dict, A_idx: int):
        """
        voyage_json: 原始 JSON dict（含 Path[]）
        A_idx:       在 Path 列表中的索引（prediction time）
        """
        # 1) 提取 A_node
        path = voyage_json['Path']
        A_node = path[A_idx]

        # 2) 构造单样本输入，逻辑同 PgETADataset.__getitem__
        from pg_dataset import build_seq_tensor, build_stat_tensor
        from pg_dataset import latlon_to_local as _ll2l

        # 2.1 A_seq, A_stat
        A_seq  = build_seq_tensor(path[:A_idx+1], A_node)
        A_stat = build_stat_tensor(voyage_json)

        # 2.2 同船历史
        mmsi = voyage_json['MMSI']
        self.cur.execute(
            "SELECT id FROM voyage_main WHERE mmsi=%s AND id<>%s "
            "LIMIT %s;", (mmsi, voyage_json['voyage_id'], self.H))
        ship_ids = [r['id'] for r in self.cur.fetchall()]
        ship_seqs = [build_seq_tensor(self._fetch_nodes(sid), A_node)
                     for sid in ship_ids]

        # 2.3 邻船集合，可根据需要调用 PgETADataset._nearby 同逻辑

        # 2.4 B_feat6
        sin_h, cos_h = encode_time(A_node['timestamp'])
        from datetime import datetime
        wd = datetime.fromisoformat(A_node['timestamp']).weekday()
        sin_w, cos_w = np.sin(2*np.pi*wd/7), np.cos(2*np.pi*wd/7)
        sin_c, cos_c = encode_time(A_node['course'])
        B6 = torch.tensor([sin_h,cos_h,sin_w,cos_w,sin_c,cos_c], dtype=torch.float)

        # 2.5 转到模型前置
        with torch.no_grad():
            A_emb    = self.embedder(A_seq.unsqueeze(0), A_stat.unsqueeze(0))
            # 类似 near_emb, ship_emb 也要做成 batch=1
            # 省略细节……

            v_pred   = self.model(B6.unsqueeze(0),
                                  A_emb, None, None, None, None)
            speed_kmh= v_pred.item()

        # 3) 路程插值：用 CubicSpline 计算从 A_idx 到终点的累计距离
        lats = np.array([pt['latitude']  for pt in path[A_idx:]])
        lons = np.array([pt['longitude'] for pt in path[A_idx:]])
        dist = np.cumsum([0] + [
            latlon_to_local(lats[i-1], lons[i-1], 0, lats[i], lons[i])[0]*self.R
            for i in range(1, len(lats))
        ])
        cs   = CubicSpline(np.arange(len(dist)), dist)
        remaining_km = cs(len(dist)-1) - cs(0)

        # 4) ETA 预测 = 当前时间 + remaining_km / speed_kmh (h)
        now = datetime.fromisoformat(A_node['timestamp'])
        delta_h = remaining_km / (speed_kmh + 1e-3)
        eta_time = now + timedelta(hours=delta_h)

        return eta_time.isoformat(), delta_h

# ———— 简单测试 ————
if __name__ == "__main__":
    with open("ship_trajectories/ship_trajectories/0_4.jsonl",'r') as f:
        v = json.load(f)[0]
    pred_eta, rem_h = ETAPredictor(
        db_params=dict(dbname="eta_voyage1", user="postgres",
                       password="y1=x2-c30", host="localhost", port=5432),
        model_ckpt="best.pth"
    )(v, len(v['Path'])//2)
    print("ETA:", pred_eta, "Remain(h):", rem_h)
