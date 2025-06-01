# train_voyage_lstm.py

import os
import glob
import json
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm


# ================================
# —— 一、定义模型：VoyageLSTMModel（方案 B：Embedding 版本）
# ================================
class VoyageLSTMModel(nn.Module):
    """
    本版本使用 Embedding 替代 One-Hot 对高基数类别进行降维。
    静态特征分为两大块：
      1) 数值特征 (num_value_dim) → 线性映射到 static_hidden 维
      2) 类别特征（Destination, Flag, StartHour, StartWeekday）依次嵌入
    序列特征经 LSTM 提取 hidden_size 维度向量后，与静态部分拼接，
    再通过全连接层输出一个实数（回归航行时长）。
    """
    def __init__(self,
                 seq_input_dim: int,
                 num_value_dim: int,
                 dest_num: int,
                 flag_num: int,
                 hour_num: int,
                 weekday_num: int,
                 hidden_size: int,
                 static_hidden: int,
                 emb_dim_dest: int = 16,
                 emb_dim_flag: int = 8,
                 emb_dim_hour: int = 4,
                 emb_dim_weekday: int = 3,
                 out_dim: int = 1):
        """
        :param seq_input_dim:   序列特征每个时间步的维度 (本例中 10)
        :param num_value_dim:   静态数值特征的列数 (本例中 29)
        :param dest_num:        Destination 类别数
        :param flag_num:        Flag 类别数
        :param hour_num:        StartHour 类别数 (大约 24)
        :param weekday_num:     StartWeekday 类别数 (大约 7)
        :param hidden_size:     LSTM 隐藏态维度
        :param static_hidden:   静态数值特征先映射到此维度
        :param emb_dim_dest:    Destination 嵌入维度
        :param emb_dim_flag:    Flag 嵌入维度
        :param emb_dim_hour:    StartHour 嵌入维度
        :param emb_dim_weekday: StartWeekday 嵌入维度
        :param out_dim:         最终回归输出维度 (一般=1)
        """
        super().__init__()

        # —— 1. LSTM 编码序列 ——
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # —— 2. 数值特征分支：29 维数值 → static_hidden
        self.fc_num_values = nn.Linear(num_value_dim, static_hidden)

        # —— 3. 类别特征 Embedding ——
        self.emb_dest     = nn.Embedding(num_embeddings=dest_num,    embedding_dim=emb_dim_dest)
        self.emb_flag     = nn.Embedding(num_embeddings=flag_num,    embedding_dim=emb_dim_flag)
        self.emb_hour     = nn.Embedding(num_embeddings=hour_num,    embedding_dim=emb_dim_hour)
        self.emb_weekday  = nn.Embedding(num_embeddings=weekday_num, embedding_dim=emb_dim_weekday)

        # —— 4. 融合后再过一个隐藏层，再输出 ——
        # 拼接后的静态向量长度 = static_hidden + emb_dim_dest + emb_dim_flag + emb_dim_hour + emb_dim_weekday
        fused_static_dim = static_hidden + emb_dim_dest + emb_dim_flag + emb_dim_hour + emb_dim_weekday
        # 融合 LSTM(hidden_size) + 静态部分(fused_static_dim) → 64 → out_dim
        self.fc1 = nn.Linear(hidden_size + fused_static_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x_seq, x_num_values, x_cat_idx):
        """
        :param x_seq:        Tensor, shape = (batch_size, T=3, seq_input_dim)
        :param x_num_values: Tensor, shape = (batch_size, num_value_dim)      (数值特征)
        :param x_cat_idx:    Tensor, shape = (batch_size, 4)                 (类别索引：dest_idx, flag_idx, hour_idx, weekday_idx)
        :return: y_pred      Tensor, shape = (batch_size,)  或  (batch_size,1)
        """
        # —— 1. LSTM 编码序列 ——
        # lstm_out: (batch, T, hidden_size),  h_n: (1, batch, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        vec_seq = h_n[-1, :, :]                     # (batch, hidden_size)

        # —— 2. 数值特征先过一个全连接，ReLU ——
        vec_num = torch.relu(self.fc_num_values(x_num_values))  # (batch, static_hidden)

        # —— 3. 类别特征 Embedding ——
        # x_cat_idx 里按顺序存放 [ dest_idx, flag_idx, hour_idx, weekday_idx ]
        dest_idx   = x_cat_idx[:, 0]   # (batch,)
        flag_idx   = x_cat_idx[:, 1]   # (batch,)
        hour_idx   = x_cat_idx[:, 2]   # (batch,)
        weekday_idx= x_cat_idx[:, 3]   # (batch,)

        vec_dest    = self.emb_dest(dest_idx)      # (batch, emb_dim_dest)
        vec_flag    = self.emb_flag(flag_idx)      # (batch, emb_dim_flag)
        vec_hour    = self.emb_hour(hour_idx)      # (batch, emb_dim_hour)
        vec_weekday = self.emb_weekday(weekday_idx) # (batch, emb_dim_weekday)

        # —— 4. 拼接所有静态分支 ——
        # static_hidden + emb_dim_dest + emb_dim_flag + emb_dim_hour + emb_dim_weekday
        combined_static = torch.cat([vec_num, vec_dest, vec_flag, vec_hour, vec_weekday], dim=1)
        # combined_static: (batch, fused_static_dim)

        # —— 5. 融合序列向量 & 静态向量 ——
        combined_all = torch.cat([vec_seq, combined_static], dim=1)  # (batch, hidden_size + fused_static_dim)
        h = torch.relu(self.fc1(combined_all))                       # (batch, 64)
        out = self.fc2(h)                                            # (batch, out_dim)
        out = out.squeeze(-1)                                        # (batch,)
        return out


# ================================
# —— 二、自定义 Dataset ——
# ================================
class VoyageDataset(Dataset):
    """
    每条样本返回：
      - X_seq       : (3, seq_feat_dim)  → LSTM 输入
      - X_num_vals  : (num_value_dim,)   → 连续数值特征
      - X_cat_idx   : (4,)               → 四个类别索引 (dest_idx, flag_idx, hour_idx, weekday_idx)
      - y           : 标量 (航行时长, hours)
    """
    def __init__(self, X_seq, X_num_vals, X_cat_idx, y):
        super().__init__()
        self.X_seq      = torch.from_numpy(X_seq).float()       # (N,3,seq_dim)
        self.X_num_vals = torch.from_numpy(X_num_vals).float()  # (N,num_value_dim)
        # 类别索引一定要是 LongTensor，才能传给 PyTorch Embedding
        self.X_cat_idx  = torch.from_numpy(X_cat_idx).long()    # (N,4)
        self.y          = torch.from_numpy(y).float().squeeze() # (N,)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (self.X_seq[idx],
                self.X_num_vals[idx],
                self.X_cat_idx[idx],
                self.y[idx])


# ================================
# —— 三、主流程：读取 JSON → 特征提取 → Embedding 预处理 → 划分数据 → 训练 & 验证
# ================================
def main():
    # —— Step 0：加载新闻数据 (predictable.csv & unpredictable.csv) ——
    base_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, '..'))

    predictable_path   = os.path.join(project_root, 'news_data', 'predictable.csv')
    unpredictable_path = os.path.join(project_root, 'news_data', 'unpredictable.csv')

    predictable_df   = pd.read_csv(predictable_path)
    unpredictable_df = pd.read_csv(unpredictable_path)

    # 确保 event_time 是 int
    predictable_df['event_time']   = predictable_df['event_time'].astype(int)
    unpredictable_df['event_time'] = unpredictable_df['event_time'].astype(int)

    # 按 event_time 升序（保险起见再排序一次）
    predictable_df   = predictable_df.sort_values('event_time').reset_index(drop=True)
    unpredictable_df = unpredictable_df.sort_values('event_time').reset_index(drop=True)

    print(f">>> Predictable 新闻条数:   {len(predictable_df)}")
    print(f">>> Unpredictable 新闻条数: {len(unpredictable_df)}\n")

    # —— Step 1：扫描并读取所有 .jsonl 文件 ——
    input_dir = os.path.join(project_root, 'ship_trajectories')
    json_paths = glob.glob(os.path.join(input_dir, '*.jsonl'))

    raw_voyages = []
    for p in tqdm(json_paths, desc="Loading JSONL files"):
        with open(p, 'r', encoding='utf-8') as f:
            # 判断文件最外层是 JSON 数组 还是 JSONL
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                # 完整 JSON 数组形式
                try:
                    arr = json.load(f)
                except json.JSONDecodeError:
                    continue
                if isinstance(arr, list):
                    raw_voyages.extend(arr)
            else:
                # 真正的 JSONL（每行一个 JSON 对象）
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw_voyages.append(obj)

    print(f"\n一共读取到 {len(raw_voyages)} 条原始航程记录（raw_voyages）。")

    # —— Step 2：遍历每条航程，抽取“静态特征”、“序列特征”及“回归标签” ——
    feature_dicts = []

    for voyage in tqdm(raw_voyages, desc="Extracting features"):
        # 要求有 Path，且至少 3 个点
        if 'Path' not in voyage or not isinstance(voyage['Path'], list) or len(voyage['Path']) < 3:
            continue

        # 2.1 计算 Start Time / End Time → delta_hours 作为回归目标
        try:
            ts_start = pd.to_datetime(voyage["Start Time"])
            ts_end   = pd.to_datetime(voyage["End Time"])
        except Exception:
            continue
        if ts_end <= ts_start:
            continue
        delta_hours = (ts_end - ts_start).total_seconds() / 3600.0  # 航行时长 (小时)

        # —— 2.2 抽取静态特征原始值 ——
        stat = {
            "MMSI":        voyage.get("MMSI", np.nan),
            "IMO":         voyage.get("IMO", np.nan),
            "Type":        voyage.get("Type", np.nan),
            "Width":       voyage.get("Width", np.nan),
            "Length":      voyage.get("Length", np.nan),
            "Flag":        voyage.get("Flag", None),
            "Destination": voyage.get("Destination", None),
            "StartTime":   ts_start,      # 保留 pd.Timestamp，后面拆 hour/weekday
            "StartLat":    voyage.get("Start Lat", np.nan),
            "StartLon":    voyage.get("Start Lon", np.nan),
            "EndLat":      voyage.get("End Lat", np.nan),
            "EndLon":      voyage.get("End Lon", np.nan),
        }

        # —— 2.3 抽取序列特征：取 Path 的前 3 个点 ——
        seq_feats = []
        for idx in range(3):
            pt = voyage["Path"][idx]
            try:
                ts_pt = pd.to_datetime(pt["timestamp"])
            except Exception:
                ts_pt = ts_start
            dt_from_start = (ts_pt - ts_start).total_seconds() / 3600.0  # 小时

            try:
                ts_eta = pd.to_datetime(pt["eta"])
                dt_eta = (ts_eta - ts_start).total_seconds() / 3600.0
            except Exception:
                dt_eta = np.nan

            seq_feats.append({
                "dt":      dt_from_start,
                "lat":     pt.get("latitude", np.nan),
                "lon":     pt.get("longitude", np.nan),
                "speed":   pt.get("speed", np.nan),
                "heading": pt.get("heading", np.nan),
                "course":  pt.get("course", np.nan),
                "draught": pt.get("draught", np.nan),
                "rot":     pt.get("rot", np.nan),
                "status":  pt.get("status", np.nan),
                "dt_eta":  dt_eta,
            })

        # —— 2.4 查找“过去新闻”（unpredictable）和“未来新闻”（predictable） ——
        cur_ts = int(ts_start.timestamp())  # 航程开始时刻（秒）

        # 不可预测新闻：event_time < cur_ts → 取最后一条
        df_pre = unpredictable_df[unpredictable_df['event_time'] < cur_ts]
        if not df_pre.empty:
            chosen_pre   = df_pre.iloc[-1]
            pre_proj_t   = chosen_pre['event_time'] - cur_ts  # 投影秒差
            pre_scores   = [chosen_pre[f'score_{i}'] for i in range(6)]
            pre_dirs     = [chosen_pre['north'], chosen_pre['south'],
                            chosen_pre['east'], chosen_pre['west']]
        else:
            pre_proj_t   = np.nan
            pre_scores   = [np.nan] * 6
            pre_dirs     = [np.nan] * 4

        # 可预测新闻：event_time > cur_ts → 取第一条
        df_post = predictable_df[predictable_df['event_time'] > cur_ts]
        if not df_post.empty:
            chosen_post   = df_post.iloc[0]
            post_proj_t   = chosen_post['event_time'] - cur_ts
            post_scores   = [chosen_post[f'score_{i}'] for i in range(6)]
            post_dirs     = [chosen_post['north'], chosen_post['south'],
                             chosen_post['east'], chosen_post['west']]
        else:
            post_proj_t   = np.nan
            post_scores   = [np.nan] * 6
            post_dirs     = [np.nan] * 4

        # —— 2.5 将这一条航程的所有信息打包到一个字典里 ——
        rec = {
            "static":      stat,
            "seq_0":       seq_feats[0],
            "seq_1":       seq_feats[1],
            "seq_2":       seq_feats[2],
            # 过去新闻特征
            "pre_proj_t":  pre_proj_t,
            "pre_scores":  pre_scores,
            "pre_dirs":    pre_dirs,
            # 未来新闻特征
            "post_proj_t": post_proj_t,
            "post_scores": post_scores,
            "post_dirs":   post_dirs,
            # 回归目标
            "y":           delta_hours
        }
        feature_dicts.append(rec)

    print(f"\n过滤后剩余 {len(feature_dicts)} 条航程样本（至少有 3 个轨迹点）。\n")
    if len(feature_dicts) == 0:
        print(">>> 没有任何符合条件的航程样本，程序退出。")
        return

    # —— Step 3：把 feature_dicts 转换为静态特征 DataFrame、序列特征 ndarray、标签向量 ——

    # 3.1. 把静态部分拆成 DataFrame
    static_records = []
    for rec in feature_dicts:
        stat = rec["static"]
        # 从 StartTime 衍生小时、星期几
        hour    = stat["StartTime"].hour
        weekday = stat["StartTime"].weekday()  # 0=周一 ... 6=周日

        d = {
            # 原来的数值静态
            "Type":         stat["Type"],
            "Width":        stat["Width"],
            "Length":       stat["Length"],
            "StartLat":     stat["StartLat"],
            "StartLon":     stat["StartLon"],
            "EndLat":       stat["EndLat"],
            "EndLon":       stat["EndLon"],

            # 类别静态 (LabelEncoder 用)
            "Destination":  stat["Destination"],
            "Flag":         stat["Flag"],
            "StartHour":    hour,
            "StartWeekday": weekday,

            # 过去新闻（数值）
            "pre_proj_t":   rec["pre_proj_t"],
            "pre_score_0":  rec["pre_scores"][0],
            "pre_score_1":  rec["pre_scores"][1],
            "pre_score_2":  rec["pre_scores"][2],
            "pre_score_3":  rec["pre_scores"][3],
            "pre_score_4":  rec["pre_scores"][4],
            "pre_score_5":  rec["pre_scores"][5],
            "pre_north":    rec["pre_dirs"][0],
            "pre_south":    rec["pre_dirs"][1],
            "pre_east":     rec["pre_dirs"][2],
            "pre_west":     rec["pre_dirs"][3],

            # 未来新闻（数值）
            "post_proj_t":  rec["post_proj_t"],
            "post_score_0": rec["post_scores"][0],
            "post_score_1": rec["post_scores"][1],
            "post_score_2": rec["post_scores"][2],
            "post_score_3": rec["post_scores"][3],
            "post_score_4": rec["post_scores"][4],
            "post_score_5": rec["post_scores"][5],
            "post_north":   rec["post_dirs"][0],
            "post_south":   rec["post_dirs"][1],
            "post_east":    rec["post_dirs"][2],
            "post_west":    rec["post_dirs"][3],
        }
        static_records.append(d)

    static_df = pd.DataFrame(static_records)

    # 3.2. 把序列特征整理成 numpy array
    #    每条样本：3 个时刻 * 10 个子特征 → 最后 reshape 成 (3,10)
    seq_keys = ["dt", "lat", "lon", "speed", "heading", "course", "draught", "rot", "status", "dt_eta"]
    all_seq = []
    for rec in feature_dicts:
        row = []
        for t in [0, 1, 2]:
            dct = rec[f"seq_{t}"]
            for k in seq_keys:
                row.append(dct.get(k, np.nan))
        all_seq.append(row)

    all_seq = np.array(all_seq, dtype=np.float32)  # (N, 3*10)
    N, flat_dim = all_seq.shape
    per_timestep = len(seq_keys)                   # 10
    all_seq = all_seq.reshape(N, 3, per_timestep)   # (N, 3, 10)

    # 3.3. 标签向量 y
    y_all = np.array([rec["y"] for rec in feature_dicts], dtype=np.float32)  # (N,)

    # —— Step 4：静态特征分两部分：数值 + LabelEncoder → Embedding ——

    # 4.1. 指定哪些列属于“类别”，哪些属于“数值”
    categorical_cols = ["Destination", "Flag", "StartHour", "StartWeekday"]
    # 剩下的一律都算数值列
    num_cols = [c for c in static_df.columns if c not in categorical_cols]

    all_nan_cols = [c for c in num_cols if static_df[c].isna().all()]
    if all_nan_cols:
        static_df.drop(columns=all_nan_cols, inplace=True)
        # 重新计算 num_cols
        num_cols = [c for c in static_df.columns if c not in categorical_cols]

    tmp = static_df[num_cols].fillna(static_df[num_cols].median())
    zero_var_cols = [col for col in num_cols if tmp[col].var() == 0.0]
    if zero_var_cols:
        static_df.drop(columns=zero_var_cols, inplace=True)
        num_cols = [c for c in static_df.columns if c not in categorical_cols]

    # —— 4.2. 对数值列做缺失值填补 & 标准化 ——
    #     填补方式：全部用中位数
    static_df[num_cols] = static_df[num_cols].fillna(static_df[num_cols].median())
    scaler_static = StandardScaler()
    num_part = scaler_static.fit_transform(static_df[num_cols])  # (N, len(num_cols)) == (N,29)

    # —— 4.3. 对类别列分别 LabelEncoder → 得到 4 列整数索引 ——
    # Destination
    static_df["Destination_filled"] = static_df["Destination"].fillna("UNKNOWN")
    le_dest = LabelEncoder()
    static_df["Destination_le"] = le_dest.fit_transform(static_df["Destination_filled"])
    dest_num = len(le_dest.classes_)  # 目的港口总数

    # Flag
    static_df["Flag_filled"] = static_df["Flag"].fillna("UNKNOWN")
    le_flag = LabelEncoder()
    static_df["Flag_le"] = le_flag.fit_transform(static_df["Flag_filled"])
    flag_num = len(le_flag.classes_)  # 船旗总数

    # StartHour
    le_hour = LabelEncoder()
    static_df["StartHour_le"] = le_hour.fit_transform(static_df["StartHour"].astype(str))
    hour_num = len(le_hour.classes_)  # 理论上 24

    # StartWeekday
    le_weekday = LabelEncoder()
    static_df["StartWeekday_le"] = le_weekday.fit_transform(static_df["StartWeekday"].astype(str))
    weekday_num = len(le_weekday.classes_)  # 理论上 7

    # 把 4 列整数组合到一起，(N,4)
    X_cat_indices = np.stack([
        static_df["Destination_le"].values,
        static_df["Flag_le"].values,
        static_df["StartHour_le"].values,
        static_df["StartWeekday_le"].values
    ], axis=1).astype(np.int64)

    print(">> 类别编码信息：")
    print(f"   Destination 共 {dest_num} 类")
    print(f"   Flag        共 {flag_num} 类")
    print(f"   StartHour   共 {hour_num} 类")
    print(f"   StartWeekday共 {weekday_num} 类\n")

    # 最终静态数值特征矩阵 (N, num_value_dim=29)
    X_num_values = num_part

    print(f"静态数值特征维度 (num_part) = {X_num_values.shape}")   # (N, 29)
    print(f"静态类别索引维度 (X_cat_indices) = {X_cat_indices.shape}")  # (N,4)

    # —— Step 5：对序列特征进行标准化 ——
    all_seq_2d = all_seq.reshape(-1, per_timestep)  # (N*3, 10)

    # (2) replace columns with all-NaN values by zeros (or drop them)
    all_nan_cols = np.isnan(all_seq_2d).all(axis=0)  # bool mask, length = per_timestep
    if all_nan_cols.any():
        print(">> Warning: the following seq features are all NaN and will be zero-filled:",
              np.array(seq_keys)[all_nan_cols].tolist())
        all_seq_2d[:, all_nan_cols] = 0.0  # or you can choose to drop these columns

    # (3) fill remaining NaNs with column-wise median
    col_median = np.nanmedian(all_seq_2d, axis=0)  # still returns NaN for all-NaN columns, but we fixed them
    inds = np.where(np.isnan(all_seq_2d))
    all_seq_2d[inds] = np.take(col_median, inds[1])

    # (4) final sanity check – no NaNs should remain
    assert not np.isnan(all_seq_2d).any(), "NaNs still present after filling!"

    scaler_seq = StandardScaler()
    all_seq_2d = scaler_seq.fit_transform(all_seq_2d)

    all_seq = all_seq_2d.reshape(N, 3, per_timestep)  # (N,3,10)
    X_seq = all_seq
    print(f"序列特征张量维度 (X_seq) = {X_seq.shape}  (N,3,10)\n")

    # —— Step 6：划分 Train / Val / Test ——
    X_seq_train,  X_seq_temp,  X_num_train,  X_num_temp,  X_cat_train,  X_cat_temp,  y_train,  y_temp = train_test_split(
        X_seq, X_num_values, X_cat_indices, y_all,
        test_size=0.30, random_state=42
    )
    X_seq_val,   X_seq_test,  X_num_val,   X_num_test,   X_cat_val,   X_cat_test,   y_val,  y_test = train_test_split(
        X_seq_temp, X_num_temp, X_cat_temp, y_temp,
        test_size=0.50, random_state=42
    )

    print(f"Train 样本数: {X_seq_train.shape[0]}")
    print(f"Val   样本数: {X_seq_val.shape[0]}")
    print(f"Test  样本数: {X_seq_test.shape[0]}\n")

    # —— Step 7：构造 Dataset & DataLoader ——
    batch_size = 32
    train_ds = VoyageDataset(X_seq_train, X_num_train, X_cat_train, y_train)
    val_ds   = VoyageDataset(X_seq_val,   X_num_val,   X_cat_val,   y_val)
    test_ds  = VoyageDataset(X_seq_test,  X_num_test,  X_cat_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # —— Step 8：实例化模型、设置损失 & 优化器 ——
    seq_input_dim    = per_timestep           # 10
    num_value_dim    = X_num_train.shape[1]   # 29
    hidden_size      = 64
    static_hidden    = 32

    model = VoyageLSTMModel(
        seq_input_dim   = seq_input_dim,
        num_value_dim   = num_value_dim,
        dest_num        = dest_num,
        flag_num        = flag_num,
        hour_num        = hour_num,
        weekday_num     = weekday_num,
        hidden_size     = hidden_size,
        static_hidden   = static_hidden,
        emb_dim_dest    = 16,
        emb_dim_flag    = 8,
        emb_dim_hour    = 4,
        emb_dim_weekday = 3,
        out_dim         = 1
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    best_val_loss = float('inf')

    # —— Step 9：训练 & 验证循环（带 tqdm 进度条） ——
    for epoch in range(1, num_epochs + 1):
        # — 9.1 训练阶段 —
        model.train()
        running_loss = 0.0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
        for X_seq_batch, X_num_batch, X_cat_batch, y_batch in train_iter:
            X_seq_batch = X_seq_batch.to(device)     # (bs, 3, 10)
            X_num_batch = X_num_batch.to(device)     # (bs, 29)
            X_cat_batch = X_cat_batch.to(device)     # (bs, 4)
            y_batch     = y_batch.to(device)         # (bs,)

            optimizer.zero_grad()
            y_pred = model(X_seq_batch, X_num_batch, X_cat_batch)  # (bs,)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_seq_batch.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # — 9.2 验证阶段 —
        model.eval()
        val_loss = 0.0

        val_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]  ", leave=False)
        with torch.no_grad():
            for X_seq_batch, X_num_batch, X_cat_batch, y_batch in val_iter:
                X_seq_batch = X_seq_batch.to(device)
                X_num_batch = X_num_batch.to(device)
                X_cat_batch = X_cat_batch.to(device)
                y_batch     = y_batch.to(device)

                y_pred = model(X_seq_batch, X_num_batch, X_cat_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_seq_batch.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch:2d}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f" Val Loss: {epoch_val_loss:.4f}")

        # 保存最优模型
        if epoch_val_loss < best_val_loss:
            print(f"[Info] 第 {epoch} 轮，验证 loss 下降：{best_val_loss:.4f} → {epoch_val_loss:.4f}，正在保存模型。")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_voyage_model.pth")

    print(f"\n训练完毕! 最佳验证 Loss = {best_val_loss:.4f}\n")

    # —— Step 10：测试集评估 ——
    model.load_state_dict(torch.load("best_voyage_model.pth"))
    model.eval()

    mse_criterion = nn.MSELoss(reduction='sum')
    mae_criterion = nn.L1Loss(reduction='sum')

    test_mse_loss = 0.0
    test_mae_loss = 0.0

    total_samples = 0

    test_iter = tqdm(test_loader, desc="Evaluating on Test Set", leave=False)
    with torch.no_grad():
        for X_seq_batch, X_num_batch, X_cat_batch, y_batch in test_iter:
            # Move data to the same device as the model
            X_seq_batch = X_seq_batch.to(device)  # (bs, 3, 10)
            X_num_batch = X_num_batch.to(device)  # (bs, 29)
            X_cat_batch = X_cat_batch.to(device)  # (bs, 4)
            y_batch = y_batch.to(device)  # (bs,)

            # Forward pass
            y_pred = model(X_seq_batch, X_num_batch, X_cat_batch)  # (bs,)

            # Compute MSE (sum over batch)
            mse_loss_batch = mse_criterion(y_pred, y_batch)
            test_mse_loss += mse_loss_batch.item()

            # Compute MAE (sum over batch)
            mae_loss_batch = mae_criterion(y_pred, y_batch)
            test_mae_loss += mae_loss_batch.item()

            total_samples += y_batch.size(0)

    # 计算平均后的 MSE 和 MAE
    test_mse = test_mse_loss / total_samples
    test_mae = test_mae_loss / total_samples

    # 打印结果
    print(f"Test MSE Loss: {test_mse:.4f}")
    print(f"Test MAE Loss: {test_mae:.4f}")


if __name__ == '__main__':
    main()
