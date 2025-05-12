"""
pg_dataset.py
──────────────────────────────────────────────────────────
Dataset 从 PostGIS 按需拉取样本，**支持多进程 DataLoader**。
"""
import math
import random
import torch
import psycopg2
import psycopg2.extras as ext
from torch.utils.data import Dataset

from utils import (
    build_seq_tensor, build_stat_tensor,
    latlon_to_local, encode_time
)
from utils import Timer, init_logger
init_logger("INFO")
# DB 连接串（请按需修改）
DB_DSN   = "dbname=eta_voyage1 user=postgres password=y1=x2-c30 host=localhost port=5432"
K_NEAR   = 10    # 邻船条数
H_SHIP   = 5     # 同船历史条数
STEP_NODE= 10    # A_seq 采样步
RADIUS_KM= 20.0  # 归一化半径

class PgETADataset(Dataset):
    def __init__(self, train: bool = True,
                 k_near=K_NEAR, h_ship=H_SHIP,
                 radius_km=RADIUS_KM, step=STEP_NODE):
        self.K, self.H, self.R, self.step = k_near, h_ship, radius_km, step
        self.train_flag = train

        # “短连接”只用来拿 ID 列表
        with psycopg2.connect(DB_DSN, cursor_factory=ext.RealDictCursor) as tmp:
            with tmp.cursor() as cur:
                flag = 'TRUE' if train else 'FALSE'
                cur.execute(f"SELECT id FROM voyage_main WHERE train={flag};")
                self.voy_ids = [r['id'] for r in cur.fetchall()]
        random.shuffle(self.voy_ids)

        # 真正用的 conn/cur 延迟到 worker 里创建
        self.conn = None
        self.cur  = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['conn'] = None
        state['cur']  = None
        return state

    def _ensure_conn(self):
        if self.conn is None:
            self.conn = psycopg2.connect(DB_DSN,
                          cursor_factory=ext.RealDictCursor)
            self.conn.autocommit = True
            self.cur  = self.conn.cursor()

    def _fetchone(self, sql, args):
        self.cur.execute(sql, args)
        return self.cur.fetchone()

    def _fetchall(self, sql, args):
        self.cur.execute(sql, args)
        return self.cur.fetchall()

    def _main(self, vid):
        return self._fetchone(
            "SELECT * FROM voyage_main WHERE id=%s;", (vid,)
        )

    def _nodes(self, vid):
        return self._fetchall(
            "SELECT * FROM voyage_node WHERE voyage_id=%s ORDER BY timestamp;",
            (vid,)
        )


    def _nearby(self, lat, lon, ex_mmsi):
        bbox_deg = self.R / 111_000.0  # 粗略把米转度 (≈1°≈111 km)

        _NEAR_SQL = f"""
        WITH ref AS (
            SELECT ST_SetSRID(ST_MakePoint(%s,%s),4326) AS p
        )
        SELECT DISTINCT ON (m.mmsi)
               n.voyage_id  AS vid,
               n.latitude, n.longitude,
               n.course,   n.speed
          FROM voyage_node  n
          JOIN voyage_main  m ON m.id = n.voyage_id
          JOIN ref r ON true
         WHERE  n.geom && ST_Expand(r.p, {bbox_deg})     -- ← ① 先过滤 BBOX
           AND ST_DWithin(n.geom, r.p, %s, true)         -- ← ② 再精确圆 + 索引
           AND m.mmsi <> %s
         ORDER BY m.mmsi,
                  n.geom <-> r.p                         -- KNN 排序
         LIMIT %s;
        """

        return self._fetchall(
            _NEAR_SQL,
            (lon, lat, self.R * 1000, ex_mmsi, self.K)
        )

    def __len__(self):
        return len(self.voy_ids)

    def __getitem__(self, idx):
        self._ensure_conn()

        vid   = self.voy_ids[idx]
        main = self._main(vid)

        nodes = self._nodes(vid)

        if len(nodes) <= self.step + 1:
            # 航迹太短，换个 idx
            return self.__getitem__((idx + 1) % len(self))

        # 随机采样 A_idx
        A_idx  = random.randrange(self.step, len(nodes)-2, self.step)
        A_node = nodes[A_idx]
        # TODO:
        B_idx = random.randrange(A_idx + 1, len(nodes))
        B_node = nodes[B_idx]

        # label = 下一节点速度 (knots→km/h)
        speed_kn = B_node.get("speed") or 0.0
        label = torch.tensor(speed_kn * 1.852, dtype=torch.float)

        # 1. A_seq & A_stat
        A_seq  = build_seq_tensor(nodes[:A_idx+1], A_node)
        A_stat = build_stat_tensor(main)

        # 2. 同船历史 H 条
        ship_rows = self._fetchall(
            "SELECT id FROM voyage_main WHERE mmsi=%s AND id<>%s "
            "ORDER BY RANDOM() LIMIT %s;",
            (main['mmsi'], vid, self.H)
        )
        ship_seqs, ship_stats = [], []
        for r in ship_rows:
            sid = r['id']
            # 生成序列和静态特征
            seq = build_seq_tensor(self._nodes(sid), A_node)
            stat = build_stat_tensor(self._main(sid))
            # 如果 seq 是空的，就给一个 (1,7) 的零占位
            if seq.numel() == 0 or seq.shape[0] == 0:
                seq = torch.zeros(1, 7, dtype=torch.float32, device=seq.device)
            # 如果 stat 是空的，也补全为长度 7 的零张量
            if stat.numel() == 0:
                stat = torch.zeros(7, dtype=torch.float32, device=stat.device)
            ship_seqs.append(seq)
            ship_stats.append(stat)
        # 不足 H 条时，再用全零占位补齐
        while len(ship_seqs) < self.H:
            ship_seqs.append(torch.zeros(1, 7, dtype=torch.float32))
            ship_stats.append(torch.zeros(7, dtype=torch.float32))

        # 3. 邻船 K 条

        near_rows = self._nearby(B_node['latitude'],
                                     B_node['longitude'],
                                     main['mmsi'])
        near_seqs, near_stats, Δxy, Δcs = [], [], [], []
        for r in near_rows:
            vid2 = r['vid']
            near_seqs.append(build_seq_tensor(self._nodes(vid2), B_node))
            near_stats.append(build_stat_tensor(self._main(vid2)))
            dx, dy = latlon_to_local(
                B_node['latitude'], B_node['longitude'],
                B_node.get('course') or 0,
                r['latitude'], r['longitude']
            )
            dtheta = math.radians(
                abs((r['course'] or 0) - (B_node.get('course') or 0))
            )
            Δxy.append([dx, dy])
            Δcs.append([math.cos(dtheta), math.sin(dtheta)])
        while len(near_seqs) < self.K:
            near_seqs.append(torch.zeros(1,7))
            near_stats.append(torch.zeros(7))
            Δxy.append([0.0, 0.0])
            Δcs.append([1.0, 0.0])

        # 4. B_feat6 —— sin/cos(hour, weekday, course)
        sin_h, cos_h = encode_time(B_node['timestamp'])

        # —— 兼容 str / datetime 的 weekday 处理 ——
        from datetime import datetime as _dt
        _ts = B_node.get('timestamp')
        if isinstance(_ts, str):
            dt = _dt.fromisoformat(_ts)
        elif isinstance(_ts, _dt):
            dt = _ts
        else:
            try:
                dt = _dt.fromisoformat(str(_ts))
            except:
                dt = _dt.utcnow()
        wd = dt.weekday()
        sin_w, cos_w = math.sin(2 * math.pi * wd / 7), math.cos(2 * math.pi * wd / 7)

        cor = B_node.get('course') or 0.0
        sin_c, cos_c = math.sin(math.radians(cor)), math.cos(math.radians(cor))

        B6 = torch.tensor([sin_h, cos_h, sin_w, cos_w, sin_c, cos_c],
                          dtype=torch.float)

        return dict(
            A_seq=A_seq,
            A_stat=A_stat,
            near_seqs=near_seqs, near_stats=near_stats,
            delta_xy=torch.tensor(Δxy,dtype=torch.float),
            delta_cs=torch.tensor(Δcs,dtype=torch.float),
            ship_seqs=ship_seqs, ship_stats=ship_stats,
            B_feat6=B6,
            label=label
        )



# collate_fn 同上，略（与之前保持一致，粘贴即可）
from utils import collate_fn
