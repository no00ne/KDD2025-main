import json
import numpy as np
from scipy.interpolate import CubicSpline
from pyproj import Transformer
import math
from datetime import datetime, timedelta
import json
import torch
import torch.nn as nn
import math
from datetime import datetime


import math
import torch
import torch.nn as nn

# ================================================================
# 1.  工具函数 —— 时间、航向、局部坐标
# ================================================================
EARTH_M = 6_371_000.0        # 地球平均半径 (m)

def encode_time(ts_str: str):
    dt = torch.tensor(float(int(ts_str[11:13])) + float(int(ts_str[14:16])) / 60.0)
    rad = 2 * math.pi * dt / 24
    return torch.sin(rad), torch.cos(rad)

def encode_course(course: float):
    rad = math.radians(course)
    return math.sin(rad), math.cos(rad)

def latlon_to_local(B_lat, B_lon, B_head_deg, lat, lon):
    """
    将 (lat,lon) 映射到 B 节点为原点、船艏向 +X 的局部坐标 (x',y')，单位 km。
    返回 tensor (Δx_km , Δy_km)
    """
    dlat = (lat - B_lat) * math.pi / 180
    dlon = (lon - B_lon) * math.pi / 180
    # 近似米制差值
    dx = EARTH_M * dlon * math.cos(math.radians(B_lat))
    dy = EARTH_M * dlat
    # 旋转到船艏坐标
    theta = math.radians(B_head_deg)
    x_p =  dx * math.cos(-theta) - dy * math.sin(-theta)
    y_p =  dx * math.sin(-theta) + dy * math.cos(-theta)
    return torch.tensor([x_p / 1000, y_p / 1000])        # km 级别

def build_B_loc_course_feat(B_lat, B_lon, B_course):
    """sin/cos(lat,lon,course) → [batch,6]"""
    lat_r, lon_r, cor_r = map(torch.deg2rad, (B_lat, B_lon, B_course))
    return torch.stack([
        torch.sin(lat_r),  torch.cos(lat_r),
        torch.sin(lon_r),  torch.cos(lon_r),
        torch.sin(cor_r),  torch.cos(cor_r)
    ], dim=-1)            # (B,6)

# ================================================================
# 2.  基础嵌入层
# ================================================================
class NodeAEmbedder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
    def forward(self, seq):                       # (B,T,7)
        _, h = self.gru(seq)
        return h[-1]                              # (B,64)

class StaticEmbedder(nn.Module):
    def __init__(self, input_dim=7, out_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, feats):                     # (B,7)
        return self.mlp(feats)                    # (B,64)

class GroupEmbedder(nn.Module):
    """一条航次 = Node 序列  + 静态信息 → 128 维"""
    def __init__(self):
        super().__init__()
        self.node = NodeAEmbedder()
        self.stat = StaticEmbedder()
    def forward(self, seq, stat):
        return torch.cat([self.node(seq), self.stat(stat)], dim=-1)  # (B,128)

# ================================================================
# 3.  注意力聚合器 & 预测网络
# ================================================================
class NearAggregator(nn.Module):
    def __init__(self, d_model=128, n_head=4):
        super().__init__()
        # 128(emb) + 4(extra) → 128
        self.proj_kv = nn.Linear(d_model + 4, d_model)
        self.attn     = nn.MultiheadAttention(d_model, n_head, batch_first=True)
    def forward(self, near_emb, near_extra, B_query):
        kv = self.proj_kv(torch.cat([near_emb, near_extra], dim=-1))
        z, w = self.attn(B_query.unsqueeze(1), kv, kv)   # (B,1,128),(B,1,K)
        return z.squeeze(1), w.squeeze(1)                # (B,128),(B,K)

class SpeedPredictor(nn.Module):
    def __init__(self, d_model=128, n_head=4):
        super().__init__()
        self.B_proj    = nn.Linear(6, d_model)
        self.near_aggr = NearAggregator(d_model, n_head)
        self.ship_attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model*4, 256), nn.ReLU(),
            nn.Linear(256, 64),        nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self,
                B_feat6,          # (B,6)
                A_emb,            # (B,128)
                near_emb,         # (B,K,128)
                near_extra,       # (B,K,4)
                ship_emb):        # (B,H,128)
        B_q = self.B_proj(B_feat6)                 # (B,128)

        near_pool, near_w = self.near_aggr(near_emb, near_extra, B_q)
        ship_z, ship_w    = self.ship_attn(B_q.unsqueeze(1), ship_emb, ship_emb)
        ship_pool = ship_z.squeeze(1)              # (B,128)

        fusion = torch.cat([B_q, A_emb, near_pool, ship_pool], dim=-1)  # (B,512)
        speed_hat = self.ffn(fusion).squeeze(-1)    # (B,)

        return speed_hat, near_w, ship_w.squeeze(1)




# 辅助函数：根据样条插值计算一段坐标序列的“曲线”长度（米）
def spline_path_length(coords, num_samples=200):
    """
    coords: [(lat, lon), ...]，至少两个点
    num_samples: 等距采样数量
    返回：路径长度，单位米
    """
    if len(coords) < 2:
        return 0.0
    # 1) 投影到平面坐标 (WebMercator 米)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xs, ys = [], []
    for lat, lon in coords:
        x, y = transformer.transform(lon, lat)
        xs.append(x)
        ys.append(y)
    # 2) 构建样条
    t = np.arange(len(xs))
    cs_x = CubicSpline(t, xs)
    cs_y = CubicSpline(t, ys)
    # 3) 采样并累加欧氏距离
    ts = np.linspace(0, len(xs) - 1, num_samples)
    sample_pts = np.vstack([cs_x(ts), cs_y(ts)]).T  # [num_samples, 2]
    deltas = sample_pts[1:] - sample_pts[:-1]
    lengths = np.linalg.norm(deltas, axis=1)
    return np.sum(lengths)

# 示例 DEMO：加载第一个航次
with open('ship_trajectories/ship_trajectories/0_4.jsonl', 'r', encoding='utf-8') as f:
    voyages = json.load(f)
voyage = voyages[0]
path = voyage['Path']

# 提取经纬度、速度、时间戳
coords, speeds, timestamps, bearings = [], [], [], []
for i, pt in enumerate(path):
    if 'latitude' in pt and 'longitude' in pt:
        lat, lon = pt['latitude'], pt['longitude']
        coords.append((lat, lon))
        speeds.append(pt.get('speed', 0.0))
        timestamps.append(datetime.fromisoformat(pt['timestamp']))
        # 计算航向
        if i > 0:
            prev_lat, prev_lon = coords[i-1]
            y = math.sin(math.radians(lon - prev_lon)) * math.cos(math.radians(lat))
            x = (math.cos(math.radians(prev_lat)) * math.sin(math.radians(lat))
                 - math.sin(math.radians(prev_lat)) * math.cos(math.radians(lat))
                   * math.cos(math.radians(lon - prev_lon)))
            bearings.append((math.degrees(math.atan2(y, x)) + 360) % 360)
        else:
            bearings.append(None)

# 确定当前节点索引
curr_idx = 100
# 计算剩余路径坐标序列
remaining_coords = coords[curr_idx:]

# 计算剩余航程长度（米），使用样条插值方法
remaining_length_m = spline_path_length(remaining_coords, num_samples= max(200, len(remaining_coords)*10))
remaining_distance_km = remaining_length_m / 1000.0

# 计算最近若干段平均航速
last_n = 50
recent_speeds = speeds[max(0, curr_idx - last_n):curr_idx] or [speeds[0]]
avg_speed_knots = sum(recent_speeds) / len(recent_speeds)
avg_speed_kmh = avg_speed_knots * 1.852

# 预测剩余时间与 ETA
if avg_speed_kmh > 0:
    remaining_time_h = remaining_distance_km / avg_speed_kmh
else:
    remaining_time_h = float('inf')
current_time = timestamps[curr_idx]
eta_pred = current_time + timedelta(hours=remaining_time_h)
actual_arrival = timestamps[curr_idx]
# 输出结果
print(f"最近 {len(recent_speeds)} 段平均航速: {avg_speed_knots:.2f} 节")
print(f"剩余航程（样条插值）: {remaining_distance_km:.2f} 公里")
print(f"预测剩余时间: {remaining_time_h:.2f} 小时")
print(f"预测到达时间 (ETA): {eta_pred}")
print(f"实际到达时间：{actual_arrival}")

# 打印速度和航向向量示例
print("\n最近 10 个节点航速 (节):", speeds[-10:])
print("最近 10 个节点航向 (度):", bearings[-10:])
