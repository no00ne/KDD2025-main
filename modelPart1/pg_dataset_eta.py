import math
import logging
import math
import os
import random
from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor    # 保留
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
# from threading import local                          # <<< REMOVE
from typing import Union

import numpy as np
import pandas as pd
# pg_dataset_eta.py 顶部（import 区域）
import psycopg2
import psycopg2.extras as ext
import psycopg2.pool as pool
import torch
from torch.utils.data import Dataset

from utils import compute_hermite_distances, get_node_related_news_tensor

logging.basicConfig(level=logging.ERROR, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

from utils import build_seq_tensor, latlon_to_local, encode_time, build_raw_seq_tensor


def _one():
    return 1.0


# DB connection settings (modify as needed)
DB_DSN = "dbname=eta_voyage2 user=cxsj host=localhost port=5433"
SPLIT_TS = 1618286640
# When using news data we further constrain the time range of training and test
# voyages.  Training voyages must end after this timestamp while test voyages
# must start before ``TEST_MAX_START_TS``.
MIN_TS = 1560700800.0
MAX_TS = 1640966400.0
K_NEAR = 32  # number of nearby vessels
H_SHIP = 10  # number of historical voyages (same vessel)
STEP_NODE = 32  # Max number of B nodes sampled from a voyage path
RADIUS_KM = 50.0  # radius for nearby search (km)
# 单例连接池：min 1, max 32 可按需调整
_PG_POOL = None
_PARENT_PID = None


def _get_pool():
    """
    如果 _PG_POOL 为 None，或当前进程 PID 与记录的 _PARENT_PID 不同，说明要么是第一次调用，
    要么是 DataLoader fork 出来的子进程，都需要在当前进程里重新 new 一个连接池。
    """
    global _PG_POOL, _PARENT_PID
    cur_pid = os.getpid()
    if _PG_POOL is None or _PARENT_PID != cur_pid:
        # new 一个只属于当前进程的新连接池
        _PG_POOL = pool.SimpleConnectionPool(minconn=1, maxconn=100,  # 根据实际场景调整最大连接数
            dsn=DB_DSN)
        _PARENT_PID = cur_pid
    return _PG_POOL


class PgETADataset(Dataset):
    # ---------- 删除 threading.local 相关逻辑 ----------
    # _thread_local = local()

    def __init__(self, train: bool = True, k_near: int = K_NEAR, h_ship: int = H_SHIP, radius_km: float = RADIUS_KM,
                 step: int = STEP_NODE,  # maximum number of B nodes sampled
                 *, m_news: int = 0,  # ⇽ 新增：新闻向量维度
                 use_news: bool = False  # ⇽ 新增：是否启用新闻特征
                 ):
        # ---------------- 基本超参 ----------------
        self.K, self.H, self.R, self.step = k_near, h_ship, radius_km, step
        self.train_flag = train
        self.m_news = int(m_news)
        self.use_news = bool(use_news and m_news > 0)  # 双条件判定

        # ---------------- 新闻数据集划分 ----------------
        if self.use_news:
            # 读取原始新闻数据
            up_news = pd.read_csv("news_data/unpredictable.csv")
            pr_news = pd.read_csv("news_data/predictable.csv")

            # 按时间戳排序
            up_news = up_news.sort_values('event_time')
            pr_news = pr_news.sort_values('event_time')

            # 计算unpredictable的划分点
            up_split_idx = int(len(up_news) * 0.8)
            up_split_time = up_news.iloc[up_split_idx]['event_time']

            # 划分unpredictable
            if train:
                self.UN_PRED_NEWS = up_news.iloc[:up_split_idx]
            else:
                self.UN_PRED_NEWS = up_news.iloc[up_split_idx:]

            # 基于unpredictable的划分时间点划分predictable
            if train:
                self.PRED_NEWS = pr_news[pr_news['event_time'] <= up_split_time]
            else:
                self.PRED_NEWS = pr_news[pr_news['event_time'] > up_split_time]
        else:
            self.UN_PRED_NEWS = None
            self.PRED_NEWS = None

        # ---------------- 流式均值 / 方差 ----------------
        self._num_mu = defaultdict(float)
        self._num_sig = defaultdict(_one)
        self._seen = 0

        # ---------------- 离散列映射 ----------------
        self.type2id = {"<UNK>": 0}
        self.flag2id = {"<UNK>": 0}
        self.port2id = {"<UNK>": 0}

        # ---------------- Load voyage id list ----------------
        with psycopg2.connect(DB_DSN, cursor_factory=ext.RealDictCursor) as tmp:
            with tmp.cursor() as cur:
                params = [SPLIT_TS]
                if train:
                    cond = "end_time < to_timestamp(%s)"
                    if self.use_news:
                        cond = f"({cond}) AND end_time > to_timestamp(%s)"
                        params.append(MIN_TS)
                else:
                    cond = "(start_time > to_timestamp(%s))"
                    if self.use_news:
                        cond = f"({cond}) AND start_time < to_timestamp(%s)"
                        params.append(MAX_TS)

                cur.execute(f"SELECT id FROM voyage_main WHERE {cond};", tuple(params), )
                self.voy_ids = [row['id'] for row in cur.fetchall()]
        random.shuffle(self.voy_ids)

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
        对于仍在进行的航次, end_* / dur_hours 可能缺失 → 按 0 处理。
        """

        # ---------- ① 数值列（8） ----------
        def z(key, default=0.0):
            val = float(main_row.get(key, default) or default)
            # 流式均值 / 方差
            self._update_running_stats(key, val)
            std = math.sqrt(max(self._num_sig[key] / max(self._seen - 1, 1e-3), 1e-6))
            return (val - self._num_mu[key]) / std

        # row 里没有就用 0 —— 这样 A_main 不会泄露未来
        num_feats = [z("width"), z("length"), z("path_len"), z("start_lat"), z("start_lon"), z("end_lat"), z("end_lon"),
            # 航次历时 (h) ; 若 end_time 缺失 ⇒ 0
            (((main_row["end_time"] - main_row["start_time"]).total_seconds() / 3600.0) if (
                        main_row.get("end_time") and main_row.get("start_time")) else 0.0)]

        # ---------- ② 类别列（3） ----------
        def idx(val, table: dict):
            if val not in table:
                table[val] = len(table)
            return float(table[val])

        cat_feats = [idx(main_row.get("type"), self.type2id), idx(main_row.get("flag"), self.flag2id),
            idx(main_row.get("end_port_wpi") or main_row.get("end_port_name"), self.port2id)]
        # ---------- ③ 起始时间周期（5） ----------
        st = main_row.get("start_time")
        if not st:
            cyc_feats = [0., 0., 0., 0., 0.]
        else:
            month = st.month - 1
            dow = st.weekday()
            hms = st.hour + st.minute / 60 + st.second / 3600
            cyc_feats = [math.sin(2 * math.pi * month / 12), math.cos(2 * math.pi * month / 12),
                math.sin(2 * math.pi * dow / 7), math.cos(2 * math.pi * dow / 7), math.sin(2 * math.pi * hms / 24)]

        return torch.tensor(num_feats + cat_feats + cyc_feats, dtype=torch.float)

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def _ensure_conn(self):
        return

    # ---------- 线程安全执行 ----------
    def _run_sql(self, sql: str, args: Union[tuple, list], fetch: str = "all"):
        """
        fetch = "all" / "one" / "none"
        每次都从连接池借连接，执行完后马上归还，避免多线程多进程时连接耗尽。
        """
        conn_pool = _get_pool()
        conn = conn_pool.getconn()  # 从连接池借
        try:
            with conn.cursor(cursor_factory=ext.RealDictCursor) as cur:

                cur.execute(sql, args)
                if fetch == "all":
                    res = cur.fetchall()
                elif fetch == "one":
                    res = cur.fetchone()
                else:
                    res = None
            return res
        finally:
            conn_pool.putconn(conn)  # 用完立刻还

    def _fetchall(self, sql, args=()):
        return self._run_sql(sql, args, "all")

    def _fetchone(self, sql, args=()):
        return self._run_sql(sql, args, "one")

    def _main(self, vid):
        return self._fetchone("SELECT * FROM voyage_main WHERE id=%s;", (vid,))

    def __del__(self):
        return

    def _nodes(self, vid):
        return self._fetchall("SELECT * FROM voyage_node WHERE voyage_id=%s ORDER BY timestamp;", (vid,))

    # pg_dataset_eta.py
    # -----------------
    # ----------------------------------------------------------------------
    #  pg_dataset_eta.py  ▸  _nearby_batch   （一次 SQL 带回 m.* 全字段）
    # ----------------------------------------------------------------------
    from datetime import datetime
    from typing import List, Dict

    def _fetch_nodes_batch(self, vid_list):
        """
        一次性从 voyage_node 表里把 vid_list 中所有 voyage_id 的节点都拉出来，
        按 voyage_id 分桶并按 timestamp 排序后返回一个 dict。
        vid_list: 可迭代航次 id 列表
        返回：{ voyage_id: [dict(row), dict(row), …] }
        """
        # 使用 PostgreSQL 的 ANY(array) 语法
        sql = """
            SELECT * 
            FROM voyage_node 
            WHERE voyage_id = ANY(%s)
            ORDER BY voyage_id, timestamp
        """
        rows = self._run_sql(sql, (list(vid_list),), fetch="all")
        nodes_map = defaultdict(list)
        for r in rows:
            nodes_map[r["voyage_id"]].append(r)

        return nodes_map

    def _nearby_batch(self, B_nodes: List[Dict],  # [{'latitude': …, 'longitude': …}, …]
            ex_mmsi: int, now: datetime ):
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
              AND  m.end_time <= %s 
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

        params = (idxs, lons, lats, ex_mmsi, now, radius_m, K)
        rows = self._fetchall(sql, params)

        # -------- 回桶 ----------
        buckets = [[] for _ in B_nodes]
        for rec in rows:
            buckets[rec["idx"]].append(rec)

        # 占位：自动按第一行字段构造，保证键齐全
        if rows:
            proto = {k: None for k in rows[0].keys() if k != "idx" and k != "dist"}
        else:  # 数据库空表时兜底
            proto = {"vid": None, "latitude": 0.0, "longitude": 0.0, "course": 0.0, "speed": 0.0, "mmsi": None}
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

        其中所有 "*"_stats_list" 都已是 **16-维张量**，方便后续 batch 堆叠。
        """
        # -------- 连接 ----------
        self._ensure_conn()
        original_idx = idx
        max_trials = len(self.voy_ids)
        trials = 0

        # 循环直到找到一个 len(nodes) > 2 的合法航次，或尝试次数达到上限
        while True:
            voyage_id = self.voy_ids[idx]
            t0 = datetime.now()
            logging.info(f"[__getitem__] 开始构造样本：voyage_id={voyage_id}, idx={idx}, time={t0}")

            # -------- 1. 读取本航次 ----------
            t1 = datetime.now()

            main = self._fetchone("SELECT * FROM voyage_main WHERE id=%s;", (voyage_id,))

            nodes = self._nodes(main['id'])
            t2 = datetime.now()
            logging.info(f"[步骤1:读取航次] voyage_id={voyage_id}, "
                         f"节点总数={len(nodes)}, 耗时={(t2 - t1).total_seconds():.3f}s")

            if len(nodes) > 2:
                # 找到合法的 nodes，跳出循环
                break

            # 否则换下一个索引继续尝试
            logging.info(f"[步骤1] 节点数不足(>2)，换下一个索引")
            trials += 1
            if trials >= max_trials:
                raise RuntimeError("所有候选航次都太短，无法构建样本。")
            idx = (idx + 1) % len(self)  # 继续循环，重新读取 voyage_id、nodes 等

        # -------- 2. 采样 A / B ----------
        t3 = datetime.now()
        # 随机选择 A 节点，保证之后至少还有一个节点可作为 B
        A_idx = random.randint(1, len(nodes) - 2)
        A_node = nodes[A_idx]
        vA = (A_node.get('speed', 0.0) or 0.0)
        speed_A = torch.tensor(vA, dtype=torch.float)

        # 构造全部 B 候选节点索引
        cand_B = list(range(A_idx + 1, len(nodes)))
        if len(cand_B) <= self.step:
            B_idxs = cand_B
        else:
            # 均匀采样 self.step 个索引
            lin = np.linspace(0, len(cand_B) - 1, self.step)
            B_idxs = [cand_B[int(round(i))] for i in lin]
        B_idxs = sorted(set(B_idxs))
        B_nodes = [nodes[i] for i in B_idxs]  # n_B 个
        t4 = datetime.now()
        logging.info(f"[步骤2:采样 A/B] voyage_id={voyage_id}, A_idx={A_idx}, "
                     f"n_B={len(B_nodes)}, 耗时={(t4 - t3).total_seconds():.3f}s")
        # -------- 3. 生成标签 (剩余小时数) ----------
        t5 = datetime.now()
        now = A_node['timestamp'] if isinstance(A_node['timestamp'], datetime) else datetime.fromisoformat(
            str(A_node['timestamp']))
        last_node = nodes[-1]
        end_time = last_node['timestamp'] if isinstance(last_node['timestamp'], datetime) else datetime.fromisoformat(
            str(last_node['timestamp']))
        label = torch.tensor((end_time - now).total_seconds() / 3600.0, dtype=torch.float)
        t6 = datetime.now()
        logging.info(f"[步骤3:生成标签] voyage_id={voyage_id}, 剩余小时数={label.item():.3f}, "
                     f"耗时={(t6 - t5).total_seconds():.3f}s")
        # -------- 4. A-序列 ----------
        t7 = datetime.now()
        A_raw = build_raw_seq_tensor(nodes[:A_idx + 1])  # (T_A,7)
        with ThreadPoolExecutor() as exe:
            A_proj_list = list(exe.map(lambda B: build_seq_tensor(nodes[:A_idx + 1], B), B_nodes))  # n_B × (T_A,7)
        t8 = datetime.now()
        logging.info(f"[步骤4:A序列构造] voyage_id={voyage_id}, T_A={A_raw.size(0)}, "
                     f"n_B={len(A_proj_list)}, 耗时={(t8 - t7).total_seconds():.3f}s")
        # A_stat：把 "未来信息" 删掉后转 16-维
        main_mask = main.copy()
        main_mask['end_time'] = None  # 不泄露终点
        main_mask.pop('dur_hours', None)
        A_stat = self._main_to_tensor(main_mask)  # (16,)
        t9 = datetime.now()
        # 4.5 计算弧长 (n_B,)
        # 取从 A_idx 开始到末尾的所有节点
        sub_nodes = nodes[A_idx:]
        speed_A_node = (nodes[A_idx].get("speed", 0.0) or 0.0)

        # 前 4 个 B 节点的索引
        first4_B_idxs = B_idxs[:4]
        # 如果 B 数量不足 4 个，就取现有的所有
        first4_B_speeds = []
        for b_ix in first4_B_idxs:
            speed_b = (nodes[b_ix].get("speed", 0.0) or 0.0)
            first4_B_speeds.append(speed_b)

        print(f"A 节点速度 (节): {speed_A_node:.3f}")
        print("前 4 个 B 节点速度 (节):", [f"{s:.3f}" for s in first4_B_speeds])
        lats_full = np.array([n['latitude'] for n in sub_nodes])
        lons_full = np.array([n['longitude'] for n in sub_nodes])
        crs_full = np.array([n.get('course', 0.0) for n in sub_nodes])
        seg_dists_nm = compute_hermite_distances(lats_full, lons_full, crs_full)

        rels = [b_idx - A_idx for b_idx in B_idxs]  # 每个 B 在 sub_nodes 中对应的位置

        # 3. 累加相邻采样点之间的距离
        dist_list = []
        prev_rel = 0  # 从 A_idx（rel=0）开始
        for r in rels:
            # seg_dists_m[prev_rel:r] 代表上一个采样点到当前采样点之间的段距离
            dist_nm = seg_dists_nm[prev_rel:r].sum()  # 单位 海里
            dist_list.append(dist_nm)
            prev_rel = r

        dist_seg = torch.tensor(dist_list, dtype=torch.float)  # (n_B,
        t10 = datetime.now()
        logging.info(f"[步骤4.5:计算弧长] voyage_id={voyage_id}, n_B={len(dist_seg)}, "
                     f"耗时={(t10 - t9).total_seconds():.3f}s")
        # ==================================================================
        # 5. 同船历史（H 条） → ship_*
        # ==================================================================
        t11 = datetime.now()

        sql_hist = ("SELECT id FROM voyage_main "
                    " WHERE mmsi=%s AND id<>%s AND end_time<%s "
                    f"ORDER BY RANDOM() LIMIT %s")

        ship_ids = [r['id'] for r in
            self._run_sql(sql_hist, (main['mmsi'], main['id'], main['start_time'], self.H), fetch="all", )]
        ship_nodes = {sid: self._nodes(sid) for sid in ship_ids}
        ship_mains = {sid: self._fetchone("SELECT * FROM voyage_main WHERE id=%s;", (sid,)) for sid in ship_ids}

        ship_raw = [build_raw_seq_tensor(ship_nodes[s]) for s in ship_ids]  # H × (T,7)
        ship_stat_vec = [self._main_to_tensor(ship_mains[s]) for s in ship_ids]  # H × (16,)

        def _proj_ship(B_ref):
            return [build_seq_tensor(ship_nodes[s], B_ref) for s in ship_ids]  # list[H]

        t12 = datetime.now()
        with ThreadPoolExecutor() as exe:
            ship_proj_list = list(exe.map(_proj_ship, B_nodes))  # n_B × H × (T,7)
        t13 = datetime.now()
        logging.info(f"[步骤5:同船历史] voyage_id={voyage_id}, n_ship={len(ship_ids)}, "
                     f"n_B={len(ship_proj_list)}, SQL+构造raw/stat共耗时={(t12 - t11).total_seconds():.3f}s, "
                     f"并发proj耗时={(t13 - t12).total_seconds():.3f}s")

        ship_raw_list = [ship_raw] * len(B_nodes)
        ship_stats_list = [ship_stat_vec] * len(B_nodes)

        # ==================================================================
        # 6. 邻船 batch 查询 → near_*
        # ==================================================================
        t14 = datetime.now()
        near_rows_list = self._nearby_batch(B_nodes, main['mmsi'], now)
        t15 = datetime.now()
        logging.info(f"[步骤6:邻船批量查询] voyage_id={voyage_id}, n_B={len(near_rows_list)}, "
                     f"耗时={(t15 - t14).total_seconds():.3f}s")
        # 6.2 收集本批次所有需要查询轨迹的 vid2（去重）
        t16 = datetime.now()
        all_vids = set()
        r_map = {}
        for rows in near_rows_list:
            for r in rows:
                vid2 = r["vid"]
                all_vids.add(vid2)
                # 如果同一个 vid 出现多次，只保留第一次的 r
                if vid2 not in r_map:
                    r_map[vid2] = r
        t17 = datetime.now()
        logging.info(f"[步骤6.2:收集 vid2] voyage_id={voyage_id}, n_vids={len(all_vids)}, "
                     f"耗时={(t17 - t16).total_seconds():.3f}s")
        # 6.3 一次性拉回所有邻船航次的节点（批量 SQL）
        t18 = datetime.now()
        nodes_map = self._fetch_nodes_batch(all_vids)
        t19 = datetime.now()
        logging.info(f"[步骤6.3:批量拉取节点] voyage_id={voyage_id}, "
                     f"航次数={len(nodes_map)}, 耗时={(t19 - t18).total_seconds():.3f}s")
        t20 = datetime.now()
        # 6.3.1 在这里再把“raw_seq_tensor”和“stat_tensor”全部预先计算好，存入两个缓存 dict
        t18b = datetime.now()
        raw_cache = {}  # vid -> build_raw_seq_tensor(nodes_map[vid])
        stat_cache = {}  # vid -> self._main_to_tensor(r_map[vid])
        for vid2 in all_vids:
            nodes2 = nodes_map.get(vid2, [])
            # 先构造“整条轨迹的( T,7 ) 张量”，并存缓存
            raw_cache[vid2] = build_raw_seq_tensor(nodes2)
            # r_map[vid2] 中既包含 voyage_main_hist 的字段，也包含当前节点字段，但是 _main_to_tensor 只会用其中的 m.* 部分
            stat_cache[vid2] = self._main_to_tensor(r_map[vid2])
        t18c = datetime.now()
        logging.info(f"[步骤6.3.1:缓存 raw/stat] voyage_id={voyage_id}, "
                     f"缓存条数={len(raw_cache)}, 耗时={(t18c - t18b).total_seconds():.3f}s")

        def _process_near(args):
            B_ref, rows, b_idx = args
            t0_b = datetime.now()
            logging.info(f"  [邻船·开始] voyage_id={voyage_id}, B_index={b_idx}, time={t0_b}")

            raw, proj, stat_vec, dxy, dcs = [], [], [], [], []
            for r in rows:
                vid2 = r["vid"]
                # 从缓存里直接取“整条航次原始轨迹 Tensor”
                seq_raw = raw_cache.get(vid2, torch.zeros(1, 7))
                raw.append(seq_raw)

                # build_seq_tensor 仍需根据 B_ref 动态计算
                proj.append(build_seq_tensor(nodes_map.get(vid2, []), B_ref))

                # 取缓存好的 stat（16 维）
                stat_vec.append(stat_cache.get(vid2, torch.zeros(16)))
                # 计算 Δx, Δy, Δcos, Δsin，仍需 r 中的经纬度、航向
                dx, dy = latlon_to_local(B_ref.get("latitude") or 0.0, B_ref.get("longitude") or 0.0,
                                         B_ref.get("course") or 0.0, r.get("latitude") or 0.0,
                                         r.get("longitude") or 0.0)
                θ = math.radians(abs((r.get("course") or 0) - (B_ref.get("course") or 0)))
                dxy.append([dx, dy])
                dcs.append([math.cos(θ), math.sin(θ)])

            while len(raw) < self.K:
                raw.append(torch.zeros(1, 7))
                proj.append(torch.zeros(1, 7))
                stat_vec.append(torch.zeros(16))
                dxy.append([0.0, 0.0])
                dcs.append([1.0, 0.0])

            t1_b = datetime.now()
            dur_b = (t1_b - t0_b).total_seconds()
            logging.info(f"  [邻船·结束] voyage_id={voyage_id}, B_index={b_idx}, "
                         f"time={t1_b}, 耗时={dur_b:.3f}s")
            return raw, proj, stat_vec, torch.tensor(dxy), torch.tensor(dcs)

        near_raw_list, near_proj_list, near_stats_list, Δxy, Δcs = [], [], [], [], []
        with ThreadPoolExecutor() as exe:
            args_iter = [(B_nodes[i], near_rows_list[i], i) for i in range(len(B_nodes))]
            for r, p, s, dx, dc in exe.map(_process_near, args_iter):
                near_raw_list.append(r)
                near_proj_list.append(p)
                near_stats_list.append(s)
                Δxy.append(dx)
                Δcs.append(dc)
        delta_xy = torch.stack(Δxy)  # (n_B, K, 2)
        delta_cs = torch.stack(Δcs)  # (n_B, K, 2)

        t21 = datetime.now()
        logging.info(f"[步骤6.4:邻船处理完毕] voyage_id={voyage_id}, "
                     f"n_B={len(near_raw_list)}, 耗时={(t21 - t20).total_seconds():.3f}s")

        # ==================================================================
        # 7. B 点 6-维周期特征
        # ==================================================================
        t17 = datetime.now()
        B6 = []
        for B in B_nodes:
            sh, ch = encode_time(B['timestamp'])
            ts = B['timestamp'] if isinstance(B['timestamp'], datetime) else datetime.fromisoformat(str(B['timestamp']))
            wd = ts.weekday()
            sw, cw = math.sin(2 * math.pi * wd / 7), math.cos(2 * math.pi * wd / 7)
            cr = B.get('course', 0.0)
            sc, cc = math.sin(math.radians(cr)), math.cos(math.radians(cr))
            B6.append(torch.tensor([sh, ch, sw, cw, sc, cc], dtype=torch.float))
        B6_list = torch.stack(B6)  # (n_B,6)
        t18 = datetime.now()
        logging.info(f"[步骤7:B6特征] voyage_id={voyage_id}, n_B={len(B6_list)}, "
                     f"耗时={(t18 - t17).total_seconds():.3f}s")
        # ==================================================================
        # 8. 新闻特征
        # ==================================================================
        t19 = datetime.now()
        news_feat = None
        if self.use_news:
            # 获取所有B节点的新闻特征
            news_feat = get_node_related_news_tensor(B_nodes, self.UN_PRED_NEWS, self.PRED_NEWS,
                                                     max_news_num=self.m_news)
        t20 = datetime.now()
        logging.info(f"[步骤8:新闻特征] voyage_id={voyage_id}, use_news={self.use_news}, "
                     f"耗时={(t20 - t19).total_seconds():.3f}s")
        # ==================================================================
        # 9. 打包返回
        # ==================================================================
        t21 = datetime.now()
        logging.info(f"[__getitem__] 完成打包，voyage_id={voyage_id}, 总耗时={(t21 - t0).total_seconds():.3f}s")
        print(f"[__getitem__] 完成打包，voyage_id={voyage_id}, 总耗时={(t21 - t0).total_seconds():.3f}s")
        del main, nodes, A_node, B_nodes, sub_nodes
        del lats_full, lons_full, crs_full, seg_dists_nm, dist_list

        del ship_ids, ship_nodes, ship_mains
        del ship_raw, ship_stat_vec

        del near_rows_list, all_vids, r_map, nodes_map
        del raw_cache, stat_cache
        del Δxy, Δcs, args_iter

        del B6, main_mask
        return {"A_raw": A_raw,  # (T_A, 7)
            "A_proj_list": A_proj_list,  # list[n_B] of (T_A,7)
            "A_stat": A_stat,  # (16,)
            # ---------- 历史同船 ----------
            "ship_raw_list": ship_raw_list,  # list[n_B] of list[H] (T,7)
            "ship_stats_list": ship_stats_list,  # list[n_B] of list[H] (16,)
            "ship_proj_list": ship_proj_list,  # list[n_B] of list[H] (T,7)
            # ---------- 邻船 ----------
            "near_raw_list": near_raw_list,  # list[n_B] of list[K] (T,7)
            "near_stats_list": near_stats_list,  # list[n_B] of list[K] (16,)
            "near_proj_list": near_proj_list,  # list[n_B] of list[K] (T,7)
            # ---------- 几何 / 时序 ----------
            "delta_xy": delta_xy,  # (n_B, K, 2)  km  ξ,η
            "delta_cs": delta_cs,  # (n_B, K, 2)  cos θ,sin θ
            "B6_list": B6_list,  # (n_B, 6)     sin/cos 周期
            "label": label,  # ()  剩余小时数 (float)
            "dist_seg": dist_seg, "speed_A": speed_A,
            "news_feat": news_feat}  # news_raw:(n_B,m,(news))  # news_proj:(n_B,m,(news))
