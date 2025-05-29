import logging
import math
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from threading import local
from typing import Union

# pg_dataset_eta.py 顶部（import 区域）
import psycopg2
import psycopg2.extras as ext
import psycopg2.pool as pool
import torch
from torch.utils.data import Dataset

from utils import build_seq_tensor, latlon_to_local, encode_time, build_raw_seq_tensor, Timer

# DB connection settings (modify as needed)
DB_DSN = "dbname=eta_voyage1 user=postgres password=y1=x2-c30 host=localhost port=5432"
K_NEAR = 32  # number of nearby vessels
H_SHIP = 10  # number of historical voyages (same vessel)
STEP_NODE = 1  # A sequence sampling step
RADIUS_KM = 50.0  # radius for nearby search (km)
# 单例连接池：min 1, max 32 可按需调整
_PG_POOL = pool.SimpleConnectionPool(
    minconn=1,
    maxconn=32,
    dsn="dbname=eta_voyage1 user=postgres password=y1=x2-c30 host=localhost port=5432"
)


class PgETADataset(Dataset):
    _thread_local = local()

    def __init__(self, train: bool = True,
                 k_near=K_NEAR, h_ship=H_SHIP,
                 radius_km=RADIUS_KM, step=STEP_NODE):
        self.K, self.H, self.R, self.step = k_near, h_ship, radius_km, step
        self.train_flag = train
        self._num_mu = defaultdict(float)  # 运行时在线更新均值 μ
        self._num_sig = defaultdict(lambda: 1.)  # 以及标准差 σ ，可先置 1 避免除 0
        self._seen = 0  # 已统计样本数（做流式均值/方差）

        # 离散列映射表
        self.type2id = {"<UNK>": 0}
        self.flag2id = {"<UNK>": 0}
        self.port2id = {"<UNK>": 0}
        # Short connection to get all voyage IDs
        with psycopg2.connect(DB_DSN, cursor_factory=ext.RealDictCursor) as tmp:
            with tmp.cursor() as cur:
                flag = 'TRUE' if train else 'FALSE'
                cur.execute(f"SELECT id FROM voyage_main WHERE train={flag};")
                self.voy_ids = [r['id'] for r in cur.fetchall()]
        random.shuffle(self.voy_ids)

        # Real connection (one per worker for multiprocessing)
        self.conn = None
        self.cur = None

    def _update_running_stats(self, key: str, val: float):
        # Welford online scheme
        self._seen += 1
        delta = val - self._num_mu[key]
        self._num_mu[key] += delta / self._seen
        delta2 = val - self._num_mu[key]
        self._num_sig[key] += delta * delta2

    # =================================================
    # =================================================
    def _main_to_tensor(self, main_row: dict) -> torch.Tensor:
        """
        voyage_main → 16-dim 数值向量
        对于仍在进行的航次，end_* / dur_hours 可能缺失 → 按 0 处理。
        """

        # ---------- ① 数值列（8） ----------
        def z(key, default=0.0):
            val = float(main_row.get(key, default) or default)
            # 流式均值 / 方差
            self._update_running_stats(key, val)
            std = math.sqrt(max(self._num_sig[key] / max(self._seen - 1, 1e-3), 1e-6))
            return (val - self._num_mu[key]) / std

        # row 里没有就用 0 —— 这样 A_main 不会泄露未来
        num_feats = [
            z("width"),
            z("length"),
            z("path_len"),
            z("start_lat"), z("start_lon"),
            z("end_lat"), z("end_lon"),
            # 航次历时 (h) ; 若 end_time 缺失 ⇒ 0
            (
                ((main_row["end_time"] - main_row["start_time"]).total_seconds() / 3600.0)
                if (main_row.get("end_time") and main_row.get("start_time")) else 0.0
            )
        ]

        # ---------- ② 类别列（3） ----------
        def idx(val, table: dict):
            if val not in table:
                table[val] = len(table)
            return float(table[val])

        cat_feats = [
            idx(main_row.get("type"), self.type2id),
            idx(main_row.get("flag"), self.flag2id),
            idx(main_row.get("end_port_wpi") or main_row.get("end_port_name"),
                self.port2id)
        ]
        # ---------- ③ 起始时间周期（5） ----------
        st = main_row.get("start_time")
        if not st:
            cyc_feats = [0., 0., 0., 0., 0.]
        else:
            month = st.month - 1
            dow = st.weekday()
            hms = st.hour + st.minute / 60 + st.second / 3600
            cyc_feats = [
                math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
                math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7),
                math.sin(2 * math.pi * hms / 24)
            ]

        return torch.tensor(num_feats + cat_feats + cyc_feats, dtype=torch.float)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['conn'] = None
        state['cur'] = None
        return state

    def _ensure_conn(self):
        if self.conn is None:
            self.conn = psycopg2.connect(DB_DSN, cursor_factory=ext.RealDictCursor)
            self.conn.autocommit = True
            self.cur = self.conn.cursor()

    # ---------- 线程安全执行 ----------
    def _run_sql(self, sql: str, args: Union[tuple, list], fetch: str = "all"):
        """
        fetch = "all" / "one" / "none"
        自动：借连接 -> 执行 -> fetch -> 归还
        """
        tl = self._thread_local
        if not hasattr(tl, "conn"):  # 首次在该线程调用
            tl.conn = _PG_POOL.getconn()  # 借一条
            tl.conn.autocommit = True
            tl.cur = tl.conn.cursor(
                cursor_factory=ext.RealDictCursor
            )

        tl.cur.execute(sql, args)
        if fetch == "all":
            res = tl.cur.fetchall()
        elif fetch == "one":
            res = tl.cur.fetchone()
        else:
            res = None
        return res

    def _fetchall(self, sql, args=()):
        return self._run_sql(sql, args, "all")

    def _fetchone(self, sql, args=()):
        return self._run_sql(sql, args, "one")

    def _main(self, vid):
        return self._fetchone(
            "SELECT * FROM voyage_main WHERE id=%s;", (vid,)
        )

    def __del__(self):
        tl = self._thread_local
        if hasattr(tl, "conn"):
            try:
                _PG_POOL.putconn(tl.conn)
            except Exception:
                pass

    def _nodes(self, vid):
        return self._fetchall(
            "SELECT * FROM voyage_node WHERE voyage_id=%s ORDER BY timestamp;",
            (vid,)
        )

    # pg_dataset_eta.py
    # -----------------
    # ----------------------------------------------------------------------
    #  pg_dataset_eta.py  ▸  _nearby_batch   （一次 SQL 带回 m.* 全字段）
    # ----------------------------------------------------------------------
    from datetime import datetime
    from typing import List, Dict

    def _nearby_batch(
            self,
            B_nodes: List[Dict],  # [{'latitude': …, 'longitude': …}, …]
            ex_mmsi: int,
            now: datetime,
    ):
        """
        对 B_nodes 中每个参考点一次性返回 K (=self.K) 条最近邻船节点。
        - 每条结果包含 voyage_node 的核心字段 + voyage_main_hist 的 **全部**列 (m.*)
        - 仍旧按原顺序回桶：返回 List[List[dict]]，len == len(B_nodes)
        """

        K = self.K
        radius_m = int(self.R * 1000)  # 搜索半径（米）

        idxs = list(range(len(B_nodes)))
        lons = [b["longitude"] for b in B_nodes]
        lats = [b["latitude"] for b in B_nodes]

        # -------- SQL：LATERAL-KNN，每点 LIMIT K --------
        sql = f"""
        WITH pts AS (
            SELECT * FROM UNNEST(
                     %s::int[]                ,  -- idx
                     %s::double precision[]   ,  -- lon
                     %s::double precision[]      -- lat
            ) AS t(idx, lon, lat)
        )
        SELECT  p.idx,
                r.*                                   -- ← n.* + m.* + dist
        FROM   pts AS p
        CROSS  JOIN LATERAL (
            SELECT n.voyage_id AS vid,
                   n.latitude,
                   n.longitude,
                   n.course,
                   n.speed,
                   m.*,                               -- <<< 所有 voyage_main_hist 列
                   n.geom <-> ST_SetSRID(
                       ST_MakePoint(p.lon,p.lat),4326) AS dist
            FROM   voyage_node       n
            JOIN   voyage_main_hist  m  ON m.id = n.voyage_id
            WHERE  m.mmsi <> %s
              {'AND m.train = TRUE' if self.train_flag else ''}
              AND  n.geom && ST_MakeEnvelope(
                       p.lon-0.6, p.lat-0.6,
                       p.lon+0.6, p.lat+0.6 , 4326)     -- 粗 BBOX
              AND  ST_DWithin(
                       n.geom,
                       ST_SetSRID(ST_MakePoint(p.lon,p.lat),4326),
                       %s, true)                        -- 半径 50 km
            ORDER  BY n.geom <-> ST_SetSRID(
                       ST_MakePoint(p.lon,p.lat),4326)  -- 纯 KNN
            LIMIT  %s
        ) AS r
        ORDER  BY p.idx, r.dist;
        """

        params = (idxs, lons, lats, ex_mmsi, radius_m, K)
        rows = self._fetchall(sql, params)

        # -------- 回桶 ----------
        buckets = [[] for _ in B_nodes]
        for rec in rows:
            buckets[rec["idx"]].append(rec)

        # 占位：自动按第一行字段构造，保证键齐全
        if rows:
            proto = {k: None for k in rows[0].keys() if k != "idx" and k != "dist"}
        else:  # 数据库空表时兜底
            proto = {
                "vid": None, "latitude": 0.0, "longitude": 0.0,
                "course": 0.0, "speed": 0.0,
                "mmsi": None
            }
        for lst in buckets:
            while len(lst) < K:
                lst.append(proto.copy())

        return buckets

    def __len__(self):
        return len(self.voy_ids)

    # ------------------------------------------------------------------
    #  PgETADataset.__getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, idx):
        """
        构造一个训练 / 验证样本并返回字典：

          ┌ A_raw                :  (T_A, 7)
          ├ A_proj_list          :  list[n_B]  each (T_A, 7)
          ├ A_stat               :  (16,)      ← _main_to_tensor(main)（去掉未来信息）
          │
          ├ ship_raw_list        :  list[n_B]  of list[H] (T,7)
          ├ ship_stats_list      :  list[n_B]  of list[H] (16,)
          ├ ship_proj_list       :  list[n_B]  of list[H] (T,7)
          │
          ├ near_raw_list        :  list[n_B]  of list[K] (T,7)
          ├ near_stats_list      :  list[n_B]  of list[K] (16,)
          ├ near_proj_list       :  list[n_B]  of list[K] (T,7)
          │
          ├ delta_xy             :  (n_B, K, 2)   km
          ├ delta_cs             :  (n_B, K, 2)   cos sin
          ├ B6_list              :  (n_B, 6)
          └ label                :  剩余小时数 (scalar)

        其中所有 “*_stats_list” 都已是 **16-维张量**，方便后续 batch 堆叠。
        """
        # -------- 连接 ----------
        self._ensure_conn()

        # -------- 1. 读取本航次 ----------
        main  = self._main(self.voy_ids[idx])
        nodes = self._nodes(main['id'])
        if len(nodes) <= self.step + 1:
            # 太短 ⇒ 换下一个 idx
            return self.__getitem__((idx + 1) % len(self))

        # -------- 2. 采样 A / B ----------
        A_idx  = random.randrange(1, len(nodes) - self.step, self.step)
        A_node = nodes[A_idx]

        B_idxs   = list(range(A_idx + self.step, len(nodes), self.step))
        B_nodes  = [nodes[i] for i in B_idxs]                # n_B 个

        # -------- 3. 生成标签 (剩余小时数) ----------
        now = A_node['timestamp'] if isinstance(A_node['timestamp'], datetime) \
              else datetime.fromisoformat(str(A_node['timestamp']))
        end_time = main['end_time'] if isinstance(main['end_time'], datetime) \
                   else datetime.fromisoformat(str(main['end_time']))
        label = torch.tensor((end_time - now).total_seconds() / 3600.0,
                             dtype=torch.float)

        # -------- 4. A-序列 ----------
        A_raw = build_raw_seq_tensor(nodes[:A_idx + 1])            # (T_A,7)
        with ThreadPoolExecutor() as exe:
            A_proj_list = list(exe.map(
                lambda B: build_seq_tensor(nodes[:A_idx + 1], B),
                B_nodes))                                          # n_B × (T_A,7)

        # A_stat：把 “未来信息” 删掉后转 16-维
        main_mask = main.copy()
        main_mask['end_time'] = None        # 不泄露终点
        main_mask.pop('dur_hours', None)
        A_stat = self._main_to_tensor(main_mask)                    # (16,)

        # ==================================================================
        # 5. 同船历史（H 条） → ship_*
        # ==================================================================
        sql_hist = (
                "SELECT id FROM voyage_main "
                " WHERE mmsi=%s AND id<>%s AND end_time<%s"
                + (" AND train=TRUE" if self.train_flag else "")
                + " ORDER BY RANDOM() LIMIT %s"
        )
        ship_ids = [r['id'] for r in self._fetchall(
            sql_hist, (main['mmsi'], main['id'], main['start_time'], self.H))]
        ship_nodes = {sid: self._nodes(sid) for sid in ship_ids}
        ship_mains = {sid: self._main(sid)  for sid in ship_ids}

        ship_raw  = [build_raw_seq_tensor(ship_nodes[s]) for s in ship_ids]        # H × (T,7)
        ship_stat_vec = [self._main_to_tensor(ship_mains[s]) for s in ship_ids]    # H × (16,)

        def _proj_ship(B_ref):
            return [build_seq_tensor(ship_nodes[s], B_ref) for s in ship_ids]      # list[H]

        with ThreadPoolExecutor() as exe:
            ship_proj_list = list(exe.map(_proj_ship, B_nodes))   # n_B × H × (T,7)

        ship_raw_list   = [ship_raw]       * len(B_nodes)
        ship_stats_list = [ship_stat_vec]  * len(B_nodes)

        # ==================================================================
        # 6. 邻船 batch 查询 → near_*
        # ==================================================================
        near_rows_list = self._nearby_batch(B_nodes, main['mmsi'], now)

        def _process_near(args):
            B_ref, rows = args
            raw, proj, stat_vec, dxy, dcs = [], [], [], [], []
            for r in rows:
                vid2 = r['vid']
                raw .append(build_raw_seq_tensor(self._nodes(vid2)))
                proj.append(build_seq_tensor(self._nodes(vid2), B_ref))
                stat_vec.append(self._main_to_tensor(r))          # (16,)
                dx, dy = latlon_to_local(
                    B_ref['latitude'], B_ref['longitude'],
                    B_ref.get('course', 0.0),
                    r['latitude'], r['longitude'])
                θ  = math.radians(abs((r.get('course') or 0) - (B_ref.get('course') or 0)))
                dxy.append([dx, dy])
                dcs.append([math.cos(θ), math.sin(θ)])
            while len(raw) < self.K:   # 占位补齐
                raw .append(torch.zeros(1,7))
                proj.append(torch.zeros(1,7))
                stat_vec.append(torch.zeros(16))
                dxy .append([0.,0.])
                dcs .append([1.,0.])
            return raw, proj, stat_vec, torch.tensor(dxy), torch.tensor(dcs)

        near_raw_list, near_proj_list, near_stats_list, Δxy, Δcs = [], [], [], [], []
        with ThreadPoolExecutor() as exe:
            for r, p, s, dx, dc in exe.map(_process_near, zip(B_nodes, near_rows_list)):
                near_raw_list .append(r)
                near_proj_list.append(p)
                near_stats_list.append(s)
                Δxy.append(dx)
                Δcs.append(dc)
        delta_xy = torch.stack(Δxy)         # (n_B,K,2)
        delta_cs = torch.stack(Δcs)         # (n_B,K,2)

        # ==================================================================
        # 7. B 点 6-维周期特征
        # ==================================================================
        B6 = []
        for B in B_nodes:
            sh, ch = encode_time(B['timestamp'])
            ts = B['timestamp'] if isinstance(B['timestamp'], datetime) \
                 else datetime.fromisoformat(str(B['timestamp']))
            wd = ts.weekday()
            sw, cw = math.sin(2*math.pi*wd/7), math.cos(2*math.pi*wd/7)
            cr = B.get('course', 0.0)
            sc, cc = math.sin(math.radians(cr)), math.cos(math.radians(cr))
            B6.append(torch.tensor([sh, ch, sw, cw, sc, cc], dtype=torch.float))
        B6_list = torch.stack(B6)            # (n_B,6)

        # ==================================================================
        # 8. 打包返回
        # ==================================================================
        return {
            "A_raw":            A_raw,              # (T_A, 7)
            "A_proj_list":      A_proj_list,        # list[n_B] of (T_A,7)
            "A_stat":           A_stat,             # (16,)  ← 已 _main_to_tensor
            # ---------- 历史同船 ----------
            "ship_raw_list":    ship_raw_list,      # list[n_B] of list[H] (T,7)
            "ship_stats_list":  ship_stats_list,    # list[n_B] of list[H] (16,)
            "ship_proj_list":   ship_proj_list,     # list[n_B] of list[H] (T,7)
            # ---------- 邻船 ----------
            "near_raw_list":    near_raw_list,      # list[n_B] of list[K] (T,7)
            "near_stats_list":  near_stats_list,    # list[n_B] of list[K] (16,)
            "near_proj_list":   near_proj_list,     # list[n_B] of list[K] (T,7)
            # ---------- 几何 / 时序 ----------
            "delta_xy":         delta_xy,           # (n_B, K, 2)  km  ξ,η
            "delta_cs":         delta_cs,           # (n_B, K, 2)  cos θ,sin θ
            "B6_list":          B6_list,            # (n_B, 6)     sin/cos 周期
            "label":            label               # ()  剩余小时数 (float)
        }    # (n_B,m,(news))


