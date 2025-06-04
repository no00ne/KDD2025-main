"""
pg_dataset_speed.py
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
K_NEAR   = 32    # 邻船条数
H_SHIP   = 10     # 同船历史条数
STEP_NODE= 5    # A_seq 采样步
RADIUS_KM= 50.0  # 归一化半径

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
        """
        构造样本：
        - 随机选择 A 节点作为“当前”时刻
        - 从 A 后以 step 间隔选取多个 B 节点
        - 对每个 B 参考点分别生成原始（raw）与局部投影（proj）三类序列：
          • A_seq
          • ship_seqs
          • near_seqs
        - 附加船舶静态特征（A_stat）和航点特征（B6）
        - label 为 A 到航次结束的剩余小时数

        返回 dict 包含：
        {
            'A_stat',
            'A_seq_raw_list', 'A_seq_proj_list',
            'ship_seqs_raw_list', 'ship_stats_raw_list', 'ship_seqs_proj_list', 'ship_stats_proj_list',
            'near_seqs_raw_list', 'near_stats_raw_list', 'near_seqs_proj_list', 'near_stats_proj_list',
            'delta_xy', 'delta_cs',
            'B6_list',
            'label'
        }
        """
        from datetime import datetime
        self._ensure_conn()
        vid = self.voy_ids[idx]
        main = self._main(vid)
        nodes = self._nodes(vid)

        # 如果轨迹太短，跳到下一个
        if len(nodes) <= self.step + 1:
            return self.__getitem__((idx + 1) % len(self))

        # 随机选 A 节点
        A_idx = random.randrange(self.step, len(nodes) - 1, self.step)
        A_node = nodes[A_idx]

        # 构造 B 节点列表
        B_idxs = list(range(A_idx + self.step, len(nodes), self.step))
        B_nodes = [nodes[i] for i in B_idxs]

        # 计算 label: A 到 end 剩余小时数
        end_time = main['end_time']
        if not isinstance(end_time, datetime):
            end_time = datetime.fromisoformat(str(end_time))
        now = datetime.fromisoformat(A_node['timestamp'])
        remain_h = (end_time - now).total_seconds() / 3600.0
        label = torch.tensor(remain_h, dtype=torch.float)

        # A 的静态特征
        A_stat = build_stat_tensor(main)

        # 初始化容器
        A_seq_raw_list, A_seq_proj_list = [], []
        ship_seqs_raw_list, ship_stats_raw_list = [], []
        ship_seqs_proj_list, ship_stats_proj_list = [], []
        near_seqs_raw_list, near_stats_raw_list = [], []
        near_seqs_proj_list, near_stats_proj_list = [], []
        delta_xy_list, delta_cs_list, B6_list = [], [], []

        # 针对每个 B_ref 构建特征
        for B_ref in B_nodes:
            # A 序列: raw 和 proj
            # raw: 保留经纬位置
            raw_feats = []
            for n in nodes[:A_idx+1]:
                spd = (n.get('speed') or 0.0)
                sin_c, cos_c = math.sin(math.radians(n.get('course',0.0))), math.cos(math.radians(n.get('course',0.0)))
                sin_h, cos_h = encode_time(n.get('timestamp',''))
                raw_feats.append([spd, sin_c, cos_c, n['latitude'], n['longitude'], sin_h, cos_h])
            A_seq_raw_list.append(torch.tensor(raw_feats, dtype=torch.float))
            A_seq_proj_list.append(build_seq_tensor(nodes[:A_idx+1], B_ref))

            # 同船历史: raw, proj 和 stats
            rows = self._fetchall(
                "SELECT id FROM voyage_main WHERE mmsi=%s AND id<>%s AND end_time<%s" +
                (" AND train=TRUE" if self.train_flag else "") +
                " ORDER BY RANDOM() LIMIT %s",
                (main['mmsi'], vid, main['start_time'], self.H)
            )
            seqs_r, stats_r, seqs_p, stats_p = [], [], [], []
            for r in rows:
                sid = r['id']
                nds = self._nodes(sid)
                # raw
                feats_r = []
                for n in nds:
                    spd = (n.get('speed') or 0.0)
                    sin_c, cos_c = math.sin(math.radians(n.get('course',0.0))), math.cos(math.radians(n.get('course',0.0)))
                    sin_h, cos_h = encode_time(n.get('timestamp',''))
                    feats_r.append([spd, sin_c, cos_c, n['latitude'], n['longitude'], sin_h, cos_h])
                seqs_r.append(torch.tensor(feats_r or [[0]*7], dtype=torch.float))
                stats_r.append(build_stat_tensor(self._main(sid)))
                # proj
                seqs_p.append(build_seq_tensor(nds, B_ref))
                stats_p.append(build_stat_tensor(self._main(sid)))
            # pad
            while len(seqs_r) < self.H:
                seqs_r.append(torch.zeros(1,7)); stats_r.append(torch.zeros(7))
                seqs_p.append(torch.zeros(1,7)); stats_p.append(torch.zeros(7))
            ship_seqs_raw_list.append(seqs_r)
            ship_stats_raw_list.append(stats_r)
            ship_seqs_proj_list.append(seqs_p)
            ship_stats_proj_list.append(stats_p)

            # 邻船: raw, proj, delta_xy, delta_cs
            near_rows = self._nearby(B_ref['latitude'], B_ref['longitude'], main['mmsi'])
            n_r, n_p, dxy, dcs = [], [], [], []
            for r in near_rows:
                nds = self._nodes(r['vid'])
                feats_nr = []
                for n in nds:
                    spd = (n.get('speed') or 0.0)
                    sin_c, cos_c = math.sin(math.radians(n.get('course',0.0))), math.cos(math.radians(n.get('course',0.0)))
                    sin_h, cos_h = encode_time(n.get('timestamp',''))
                    feats_nr.append([spd, sin_c, cos_c, n['latitude'], n['longitude'], sin_h, cos_h])
                n_r.append(torch.tensor(feats_nr or [[0]*7], dtype=torch.float))
                n_p.append(build_seq_tensor(nds, B_ref))
                dx, dy = latlon_to_local(B_ref['latitude'], B_ref['longitude'], B_ref.get('course',0.0), r['latitude'], r['longitude'])
                theta = math.radians(abs((r.get('course',0.0) - B_ref.get('course',0.0))))
                dxy.append([dx, dy]); dcs.append([math.cos(theta), math.sin(theta)])
            while len(n_r) < self.K:
                n_r.append(torch.zeros(1,7)); n_p.append(torch.zeros(1,7)); dxy.append([0.0,0.0]); dcs.append([1.0,0.0])
            near_seqs_raw_list.append(n_r)
            near_stats_raw_list.append([build_stat_tensor(self._main(r['vid'])) for r in near_rows] + [[0]*7]*(self.K-len(near_rows)))
            near_seqs_proj_list.append(n_p)
            near_stats_proj_list.append([build_stat_tensor(self._main(r['vid'])) for r in near_rows] + [[0]*7]*(self.K-len(near_rows)))
            delta_xy_list.append(torch.tensor(dxy, dtype=torch.float))
            delta_cs_list.append(torch.tensor(dcs, dtype=torch.float))

            # B6 特征
            sin_h, cos_h = encode_time(B_ref['timestamp'])
            wd = datetime.fromisoformat(B_ref['timestamp']).weekday()
            sin_w, cos_w = math.sin(2*math.pi*wd/7), math.cos(2*math.pi*wd/7)
            cr = B_ref.get('course',0.0)
            sin_c, cos_c = math.sin(math.radians(cr)), math.cos(math.radians(cr))
            B6_list.append(torch.tensor([sin_h,cos_h,sin_w,cos_w,sin_c,cos_c], dtype=torch.float))

        return {
            'A_stat': A_stat,
            'A_seq_raw_list':    A_seq_raw_list,
            'A_seq_proj_list':   A_seq_proj_list,
            'ship_seqs_raw_list':    ship_seqs_raw_list,
            'ship_stats_raw_list':   ship_stats_raw_list,
            'ship_seqs_proj_list':   ship_seqs_proj_list,
            'ship_stats_proj_list':  ship_stats_proj_list,
            'near_seqs_raw_list':    near_seqs_raw_list,
            'near_stats_raw_list':   near_stats_raw_list,
            'near_seqs_proj_list':   near_seqs_proj_list,
            'near_stats_proj_list':  near_stats_proj_list,
            'delta_xy':          torch.stack(delta_xy_list),
            'delta_cs':          torch.stack(delta_cs_list),
            'B6_list':           torch.stack(B6_list),
            'label':             label
        }




# collate_fn 同上，略（与之前保持一致，粘贴即可）
from utils import collate_fn_speed
