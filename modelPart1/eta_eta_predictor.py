# eta_eta_predictor.py
import math, json, psycopg2
from datetime import datetime, timedelta

import numpy as np
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn

# --- 引入前面写好的子网络 --------------------------
from eta_speed_model import GroupEmbedder, SpeedPredictor, encode_time  # ← 上一步生成的文件
from eta_speed_model import build_B_loc_course_feat, latlon_to_local
# --------------------------------------------------

# ================= 工具函数 =========================
EARTH_R = 6371.0  # km

def haversine_km(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return 2 * EARTH_R * math.asin(math.sqrt(a))

def smooth_path(path_lats, path_lons, step_km=2.0):
    """将原始节点 (lat,lon) 用立方样条平滑并按 ~step_km 采样"""
    # 1) 累积弧长 (粗略)
    s = [0.0]
    for i in range(1, len(path_lats)):
        s.append(s[-1] + haversine_km(path_lats[i-1], path_lons[i-1],
                                      path_lats[i],   path_lons[i]))
    s = np.array(s)
    # 2) 样条
    lat_spl = CubicSpline(s, path_lats)
    lon_spl = CubicSpline(s, path_lons)
    # 3) 均匀采样
    new_s = np.arange(0, s[-1], step_km)
    return lat_spl(new_s), lon_spl(new_s)

# =============== 数据库访问类 =======================
class VoyageDB:
    def __init__(self, db_params):
        self.conn = psycopg2.connect(**db_params)
    # ---- 同船历史 H 条 -----------------------------
    def fetch_ship_history(self, mmsi, limit=5):
        cur = self.conn.cursor()
        cur.execute("""
            SELECT v.*, n.* FROM voyage_main v
            JOIN voyage_node n ON n.voyage_id = v.id
            WHERE v.mmsi = %s
            ORDER BY v.start_time DESC
            LIMIT %s;
        """, (mmsi, limit))
        rows = cur.fetchall()
        return self._rows_to_voyages(rows)

    # ---- 邻域 K 条 ---------------------------------
    def fetch_nearby(self, lat, lon, course_deg, radius_km=20, limit=10):
        cur = self.conn.cursor()
        cur.execute("""
          SELECT v.*, n.*
          FROM voyage_node n
          JOIN voyage_main v ON v.id = n.voyage_id
          WHERE ST_DWithin(
              n.geom,
              ST_SetSRID(ST_MakePoint(%s,%s),4326)::geography,
              %s)
          ORDER BY ST_Distance(
              n.geom,
              ST_SetSRID(ST_MakePoint(%s,%s),4326))
          LIMIT %s;
        """, (lon, lat, radius_km*1000, lon, lat, limit))
        rows = cur.fetchall()
        return self._rows_to_voyages(rows)

    # ------------- 把数据库行转回 “{Path:[],…}” -------------
    def _rows_to_voyages(self, rows):
        voyages = {}
        for row in rows:
            vid = row[0]                         # voyage_main.id
            if vid not in voyages:
                voyages[vid] = {
                    "MMSI": row[1], "Width": row[6], "Length": row[7],
                    "Path Len": row[15],
                    "Start Lat": row[12], "Start Lon": row[13],
                    "End Lat": row[14],   "End Lon": row[15],
                    "Path": []
                }
            # 节点列索引根据 SELECT 顺序自己改；这里只示意
            voyages[vid]["Path"].append({
                "timestamp": row[-9],
                "latitude":  row[-8],
                "longitude": row[-7],
                "speed":     row[-4],
                "heading":   row[-3],
                "course":    row[-2],
            })
        return list(voyages.values())

# ================= ETA 预测包装器 ===================
class ETAPredictor(nn.Module):
    def __init__(self,
                 db_params,
                 K=10, H=5,
                 radius_km=20,
                 smooth_step=2.0):
        super().__init__()
        self.db = VoyageDB(db_params)
        self.group = GroupEmbedder()
        self.speed = SpeedPredictor()
        self.K = K; self.H = H
        self.radius_km = radius_km
        self.smooth_step = smooth_step

    # --------- 入口函数 ----------
    def forward(self, voyage:dict, A_idx:int):
        path = voyage["Path"]
        A_node = path[A_idx]
        now_time = datetime.fromisoformat(A_node["timestamp"])

        # 1. A_emb
        seq_A  = self._build_seq(path[:A_idx+1])
        stat_A = self._build_stat(voyage)
        A_emb  = self.group(seq_A, stat_A)          # (1,128)

        # 2. ship_emb (B,H,128)
        ship_voys = self.db.fetch_ship_history(voyage["MMSI"], self.H)
        ship_emb  = self._embed_many(ship_voys)     # (1,H,128)

        # 3. 平滑未来轨迹 lat/lon (含 A 点)
        fut_lats = [pt["latitude"] for pt in path[A_idx:]]
        fut_lons = [pt["longitude"] for pt in path[A_idx:]]
        lat_s, lon_s = smooth_path(fut_lats, fut_lons, self.smooth_step)

        total_sec = 0.0
        prev_lat, prev_lon = A_node["latitude"], A_node["longitude"]
        prev_speed = A_node.get("speed", 10.0)      # 若原速缺失

        for j in range(1, len(lat_s)):              # 每个 B=lat_s[j]
            B_lat, B_lon = float(lat_s[j]), float(lon_s[j])
            # --- B_feat6 ---
            # 若无航向，用邻点估算
            if j+1 < len(lat_s):
                bearing = math.degrees(math.atan2(
                    (lon_s[j+1]-B_lon)*math.cos(math.radians(B_lat)),
                    (lat_s[j+1]-B_lat)))
            else:
                bearing = voyage["Path"][-1].get("course", 0.0)
            B_feat6 = build_B_loc_course_feat(
                torch.tensor([B_lat]), torch.tensor([B_lon]),
                torch.tensor([bearing]))            # (1,6)

            # --- near_emb ---
            near_voys = self.db.fetch_nearby(B_lat, B_lon, bearing,
                                             self.radius_km, self.K)
            if near_voys:
                near_emb = self._embed_many(near_voys)         # (1,K,128)
                # Δx′、Δy′、Δθ
                extras   = []
                for nv in near_voys:
                    nlat = nv["Path"][0]["latitude"]
                    nlon = nv["Path"][0]["longitude"]
                    ex,ey = latlon_to_local(B_lat, B_lon, bearing,
                                            nlat, nlon)
                    d_theta = math.radians(
                        abs(nv["Path"][0].get("course",0)-bearing))
                    extras.append([ex, ey, math.cos(d_theta), math.sin(d_theta)])
                near_extra = torch.tensor(extras).unsqueeze(0)  # (1,K,4)
            else:                                              # 无邻居兜底
                near_emb   = torch.zeros(1,1,128)
                near_extra = torch.zeros(1,1,4)

            # --- 调速模型 ---
            speed_hat,_,_ = self.speed(B_feat6, A_emb,
                                       near_emb, near_extra, ship_emb)
            v_kmh = float(speed_hat.item()) if speed_hat.item()>1e-3 else 5.0
            # --- 距离 & 时间 ---
            dist = haversine_km(prev_lat, prev_lon, B_lat, B_lon)   # km
            seg_sec = dist / v_kmh * 3600
            total_sec += seg_sec
            prev_lat, prev_lon = B_lat, B_lon
            prev_speed = v_kmh

        eta = now_time + timedelta(seconds=total_sec)
        return eta, total_sec/3600   # 返回 ETA 时间和剩余小时数

    # --------- 内部工具 ----------
    def _build_seq(self, path_slice):
        feats = []
        for pt in path_slice:
            s = pt.get("speed", 0.0)
            sin_c,cos_c = math.sin(math.radians(pt.get("course",0))), \
                          math.cos(math.radians(pt.get("course",0)))
            sin_h,cos_h = encode_time(pt["timestamp"])
            feats.append([s,sin_c,cos_c,
                          pt["latitude"],pt["longitude"],
                          sin_h,cos_h])
        return torch.tensor(feats).unsqueeze(0).float()

    def _build_stat(self, voy):
        feats=[voy.get('Width',0), voy.get('Length',0), voy.get('Path Len',0),
               voy.get('Start Lat',0), voy.get('Start Lon',0),
               voy.get('End Lat',0),   voy.get('End Lon',0)]
        return torch.tensor([feats]).float()

    def _embed_many(self, voyages):
        if not voyages:                                  # 防止空 list
            return torch.zeros(1,1,128)
        seqs=[]; stats=[]
        for v in voyages:
            seqs.append(self._build_seq(v["Path"]))
            stats.append(self._build_stat(v))
        seqs  = torch.cat(seqs,  dim=0)   # (N,T,7)
        stats = torch.cat(stats, dim=0)   # (N,7)
        emb   = self.group(seqs, stats)   # (N,128)
        # reshape 为 (1,N,128)
        return emb.unsqueeze(0)

# ================= 简单测试 =========================
if __name__ == "__main__":
    with open("ship_trajectories/ship_trajectories/0_4.jsonl","r",encoding="utf-8") as f:
        voyage = json.load(f)[0]
    A_index = len(voyage["Path"])//2

    eta_model = ETAPredictor(
        db_params = dict(dbname="eta_voyage1", user="postgres",
                         password="y1=x2-c30", host="localhost", port=5432),
        K=10, H=5, radius_km=30)

    eta_time, remain_hr = eta_model(voyage, A_index)
    print("预测 ETA :", eta_time)
    print("剩余小时 :", remain_hr)
