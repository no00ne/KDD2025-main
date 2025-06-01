# train_voyage_lstm_no_news.py

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
# —— 1. 定义模型：VoyageLSTMModel（方案 B：Embedding 版本） ——
# ================================
class VoyageLSTMModel(nn.Module):
    """
    This version uses embeddings for high-cardinality categorical features
    instead of one-hot encoding. Static features are split into:
      1) Numerical features (num_value_dim) → linear layer to static_hidden
      2) Categorical features (Destination, Flag, StartHour, StartWeekday) → embeddings
    Sequence features go through an LSTM to produce a hidden vector (hidden_size),
    then concatenated with the static part and passed through fully connected layers
    to output a single regression value (voyage duration).
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
        :param seq_input_dim:   Dimension of sequence features per time step (here 10)
        :param num_value_dim:   Number of numerical static features (here 7)
        :param dest_num:        Number of classes for Destination
        :param flag_num:        Number of classes for Flag
        :param hour_num:        Number of classes for StartHour (≈24)
        :param weekday_num:     Number of classes for StartWeekday (≈7)
        :param hidden_size:     LSTM hidden size
        :param static_hidden:   Hidden size for numerical static branch
        :param emb_dim_dest:    Embedding dimension for Destination
        :param emb_dim_flag:    Embedding dimension for Flag
        :param emb_dim_hour:    Embedding dimension for StartHour
        :param emb_dim_weekday: Embedding dimension for StartWeekday
        :param out_dim:         Final output dimension (usually 1)
        """
        super().__init__()

        # 1. LSTM for sequence encoding
        self.lstm = nn.LSTM(
            input_size=seq_input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )

        # 2. Numerical static branch: num_value_dim → static_hidden
        self.fc_num_values = nn.Linear(num_value_dim, static_hidden)

        # 3. Embeddings for categorical static features
        self.emb_dest     = nn.Embedding(num_embeddings=dest_num,    embedding_dim=emb_dim_dest)
        self.emb_flag     = nn.Embedding(num_embeddings=flag_num,    embedding_dim=emb_dim_flag)
        self.emb_hour     = nn.Embedding(num_embeddings=hour_num,    embedding_dim=emb_dim_hour)
        self.emb_weekday  = nn.Embedding(num_embeddings=weekday_num, embedding_dim=emb_dim_weekday)

        # 4. Fusion layer: (hidden_size + fused_static_dim) → 64 → out_dim
        fused_static_dim = static_hidden + emb_dim_dest + emb_dim_flag + emb_dim_hour + emb_dim_weekday
        self.fc1 = nn.Linear(hidden_size + fused_static_dim, 64)
        self.fc2 = nn.Linear(64, out_dim)

    def forward(self, x_seq, x_num_values, x_cat_idx):
        """
        :param x_seq:        Tensor, shape = (batch_size, T=3, seq_input_dim)
        :param x_num_values: Tensor, shape = (batch_size, num_value_dim)
        :param x_cat_idx:    Tensor, shape = (batch_size, 4)  (dest_idx, flag_idx, hour_idx, weekday_idx)
        :return y_pred       Tensor, shape = (batch_size,) or (batch_size,1)
        """
        # 1. LSTM encoding for sequence
        lstm_out, (h_n, c_n) = self.lstm(x_seq)
        vec_seq = h_n[-1, :, :]  # (batch, hidden_size)

        # 2. Numerical static branch → ReLU
        vec_num = torch.relu(self.fc_num_values(x_num_values))  # (batch, static_hidden)

        # 3. Embeddings for categorical features
        dest_idx    = x_cat_idx[:, 0]  # (batch,)
        flag_idx    = x_cat_idx[:, 1]  # (batch,)
        hour_idx    = x_cat_idx[:, 2]  # (batch,)
        weekday_idx = x_cat_idx[:, 3]  # (batch,)

        vec_dest    = self.emb_dest(dest_idx)     # (batch, emb_dim_dest)
        vec_flag    = self.emb_flag(flag_idx)     # (batch, emb_dim_flag)
        vec_hour    = self.emb_hour(hour_idx)     # (batch, emb_dim_hour)
        vec_weekday = self.emb_weekday(weekday_idx)  # (batch, emb_dim_weekday)

        # 4. Concatenate all static branches
        combined_static = torch.cat([vec_num, vec_dest, vec_flag, vec_hour, vec_weekday], dim=1)
        # combined_static: (batch, fused_static_dim)

        # 5. Fuse sequence vector & static vector
        combined_all = torch.cat([vec_seq, combined_static], dim=1)  # (batch, hidden_size + fused_static_dim)
        h = torch.relu(self.fc1(combined_all))                       # (batch, 64)
        out = self.fc2(h)                                            # (batch, out_dim)
        out = out.squeeze(-1)                                        # (batch,)
        return out


# ================================
# —— 2. 自定义 Dataset ——
# ================================
class VoyageDataset(Dataset):
    """
    Each sample returns:
      - X_seq       : (3, seq_feat_dim)  → input to LSTM
      - X_num_vals  : (num_value_dim,)   → continuous numerical static features
      - X_cat_idx   : (4,)               → four categorical indices (dest_idx, flag_idx, hour_idx, weekday_idx)
      - y           : scalar (voyage duration in hours)
    """
    def __init__(self, X_seq, X_num_vals, X_cat_idx, y):
        super().__init__()
        self.X_seq      = torch.from_numpy(X_seq).float()       # (N, 3, seq_dim)
        self.X_num_vals = torch.from_numpy(X_num_vals).float()  # (N, num_value_dim)
        self.X_cat_idx  = torch.from_numpy(X_cat_idx).long()    # (N, 4)
        self.y          = torch.from_numpy(y).float().squeeze() # (N,)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        return (self.X_seq[idx],
                self.X_num_vals[idx],
                self.X_cat_idx[idx],
                self.y[idx])


# ================================
# —— 3. 主流程：读取 JSON → 特征提取 → 预处理 → 划分数据 → 训练 & 验证 ——
# ================================
def main():
    # —— Step 1：扫描并读取所有 .jsonl 文件 ——
    base_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(base_dir, '..'))

    input_dir = os.path.join(project_root, 'ship_trajectories')
    json_paths = glob.glob(os.path.join(input_dir, '*.jsonl'))

    raw_voyages = []
    for p in tqdm(json_paths, desc="Loading JSONL files"):
        with open(p, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            if first_char == '[':
                # JSON array format
                try:
                    arr = json.load(f)
                except json.JSONDecodeError:
                    continue
                if isinstance(arr, list):
                    raw_voyages.extend(arr)
            else:
                # JSONL format (one JSON object per line)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw_voyages.append(obj)

    print(f"\nLoaded {len(raw_voyages)} raw voyages.\n")

    # —— Step 2：遍历每条航程，抽取静态特征、序列特征及回归标签 ——
    feature_dicts = []

    for voyage in tqdm(raw_voyages, desc="Extracting features"):
        # Require 'Path' with at least 3 points
        if 'Path' not in voyage or not isinstance(voyage['Path'], list) or len(voyage['Path']) < 3:
            continue

        # 2.1 Compute Start Time / End Time → delta_hours as target
        try:
            ts_start = pd.to_datetime(voyage["Start Time"])
            ts_end   = pd.to_datetime(voyage["End Time"])
        except Exception:
            continue
        if ts_end <= ts_start:
            continue
        delta_hours = (ts_end - ts_start).total_seconds() / 3600.0  # voyage duration in hours

        # 2.2 Extract static raw features
        stat = {
            "MMSI":        voyage.get("MMSI", np.nan),
            "IMO":         voyage.get("IMO", np.nan),
            "Type":        voyage.get("Type", np.nan),
            "Width":       voyage.get("Width", np.nan),
            "Length":      voyage.get("Length", np.nan),
            "Flag":        voyage.get("Flag", None),
            "Destination": voyage.get("Destination", None),
            "StartTime":   ts_start,      # keep pandas.Timestamp, will split to hour/weekday later
            "StartLat":    voyage.get("Start Lat", np.nan),
            "StartLon":    voyage.get("Start Lon", np.nan),
            "EndLat":      voyage.get("End Lat", np.nan),
            "EndLon":      voyage.get("End Lon", np.nan),
        }

        # 2.3 Extract sequence features: take first 3 points from 'Path'
        seq_feats = []
        for idx in range(3):
            pt = voyage["Path"][idx]
            try:
                ts_pt = pd.to_datetime(pt["timestamp"])
            except Exception:
                ts_pt = ts_start
            dt_from_start = (ts_pt - ts_start).total_seconds() / 3600.0  # hours

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

        # 2.4 Package features into one record
        rec = {
            "static": stat,
            "seq_0":  seq_feats[0],
            "seq_1":  seq_feats[1],
            "seq_2":  seq_feats[2],
            "y":      delta_hours
        }
        feature_dicts.append(rec)

    print(f"\nAfter filtering, {len(feature_dicts)} voyages remain (each has ≥ 3 trajectory points).\n")
    if len(feature_dicts) == 0:
        print(">>> No valid voyage samples, exiting.")
        return

    # —— Step 3：转换 feature_dicts → static DataFrame, sequence ndarray, label vector ——

    # 3.1 Build static DataFrame
    static_records = []
    for rec in feature_dicts:
        stat = rec["static"]
        # Derive hour and weekday from StartTime
        hour    = stat["StartTime"].hour
        weekday = stat["StartTime"].weekday()  # 0=Monday ... 6=Sunday

        d = {
            # Original numerical static features
            "Type":        stat["Type"],
            "Width":       stat["Width"],
            "Length":      stat["Length"],
            "StartLat":    stat["StartLat"],
            "StartLon":    stat["StartLon"],
            "EndLat":      stat["EndLat"],
            "EndLon":      stat["EndLon"],

            # Categorical static (for LabelEncoder)
            "Destination": stat["Destination"],
            "Flag":        stat["Flag"],
            "StartHour":   hour,
            "StartWeekday": weekday,
        }
        static_records.append(d)

    static_df = pd.DataFrame(static_records)

    # 3.2 Build sequence ndarray
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

    # 3.3 Label vector y
    y_all = np.array([rec["y"] for rec in feature_dicts], dtype=np.float32)  # (N,)

    # —— Step 4：静态特征预处理：数值 + LabelEncoder → Embedding ——

    # 4.1 Specify categorical vs. numerical columns
    categorical_cols = ["Destination", "Flag", "StartHour", "StartWeekday"]
    num_cols = [c for c in static_df.columns if c not in categorical_cols]

    # Drop columns that are all NaN
    all_nan_cols = [c for c in num_cols if static_df[c].isna().all()]
    if all_nan_cols:
        static_df.drop(columns=all_nan_cols, inplace=True)
        num_cols = [c for c in static_df.columns if c not in categorical_cols]

    # Remove zero-variance columns
    tmp = static_df[num_cols].fillna(static_df[num_cols].median())
    zero_var_cols = [col for col in num_cols if tmp[col].var() == 0.0]
    if zero_var_cols:
        static_df.drop(columns=zero_var_cols, inplace=True)
        num_cols = [c for c in static_df.columns if c not in categorical_cols]

    # 4.2 Fill missing values and standardize numerical columns
    static_df[num_cols] = static_df[num_cols].fillna(static_df[num_cols].median())
    scaler_static = StandardScaler()
    num_part = scaler_static.fit_transform(static_df[num_cols])  # (N, len(num_cols))

    # 4.3 Label encode categorical columns to integer indices
    # Destination
    static_df["Destination_filled"] = static_df["Destination"].fillna("UNKNOWN")
    le_dest = LabelEncoder()
    static_df["Destination_le"] = le_dest.fit_transform(static_df["Destination_filled"])
    dest_num = len(le_dest.classes_)  # number of unique destinations

    # Flag
    static_df["Flag_filled"] = static_df["Flag"].fillna("UNKNOWN")
    le_flag = LabelEncoder()
    static_df["Flag_le"] = le_flag.fit_transform(static_df["Flag_filled"])
    flag_num = len(le_flag.classes_)  # number of unique flags

    # StartHour
    le_hour = LabelEncoder()
    static_df["StartHour_le"] = le_hour.fit_transform(static_df["StartHour"].astype(str))
    hour_num = len(le_hour.classes_)  # ≈24

    # StartWeekday
    le_weekday = LabelEncoder()
    static_df["StartWeekday_le"] = le_weekday.fit_transform(static_df["StartWeekday"].astype(str))
    weekday_num = len(le_weekday.classes_)  # ≈7

    # Stack the 4 integer-encoded columns into (N, 4)
    X_cat_indices = np.stack([
        static_df["Destination_le"].values,
        static_df["Flag_le"].values,
        static_df["StartHour_le"].values,
        static_df["StartWeekday_le"].values
    ], axis=1).astype(np.int64)

    print(">> Categorical encoding info:")
    print(f"   Destination classes: {dest_num}")
    print(f"   Flag        classes: {flag_num}")
    print(f"   StartHour   classes: {hour_num}")
    print(f"   StartWeekday classes: {weekday_num}\n")

    # Final numerical static feature matrix (N, num_value_dim)
    X_num_values = num_part
    print(f"Numerical static features shape (X_num_values) = {X_num_values.shape}")
    print(f"Categorical indices shape    (X_cat_indices) = {X_cat_indices.shape}\n")

    # —— Step 5：对序列特征进行标准化 ——
    all_seq_2d = all_seq.reshape(-1, per_timestep)  # (N*3, 10)

    # Replace columns that are all-NaN with zeros
    all_nan_cols = np.isnan(all_seq_2d).all(axis=0)
    if all_nan_cols.any():
        print(">> Warning: the following seq features are all NaN and will be zero-filled:",
              np.array(seq_keys)[all_nan_cols].tolist())
        all_seq_2d[:, all_nan_cols] = 0.0

    # Fill remaining NaNs with column-wise median
    col_median = np.nanmedian(all_seq_2d, axis=0)
    inds = np.where(np.isnan(all_seq_2d))
    all_seq_2d[inds] = np.take(col_median, inds[1])

    # Ensure no NaNs remain
    assert not np.isnan(all_seq_2d).any(), "NaNs still present after filling!"

    scaler_seq = StandardScaler()
    all_seq_2d = scaler_seq.fit_transform(all_seq_2d)

    all_seq = all_seq_2d.reshape(N, 3, per_timestep)  # (N, 3, 10)
    X_seq = all_seq
    print(f"Sequence features shape (X_seq) = {X_seq.shape}  (N, 3, 10)\n")

    # —— Step 6：划分 Train / Val / Test ——
    X_seq_train,  X_seq_temp,  X_num_train,  X_num_temp,  X_cat_train,  X_cat_temp,  y_train,  y_temp = train_test_split(
        X_seq, X_num_values, X_cat_indices, y_all,
        test_size=0.30, random_state=42
    )
    X_seq_val,   X_seq_test,  X_num_val,   X_num_test,   X_cat_val,   X_cat_test,   y_val,  y_test = train_test_split(
        X_seq_temp, X_num_temp, X_cat_temp, y_temp,
        test_size=0.50, random_state=42
    )

    print(f"Train samples: {X_seq_train.shape[0]}")
    print(f"Val   samples: {X_seq_val.shape[0]}")
    print(f"Test  samples: {X_seq_test.shape[0]}\n")

    # —— Step 7：构造 Dataset & DataLoader ——
    batch_size = 32
    train_ds = VoyageDataset(X_seq_train, X_num_train, X_cat_train, y_train)
    val_ds   = VoyageDataset(X_seq_val,   X_num_val,   X_cat_val,   y_val)
    test_ds  = VoyageDataset(X_seq_test,  X_num_test,  X_cat_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # —— Step 8：实例化模型、设置损失 & 优化器 ——
    seq_input_dim  = per_timestep           # 10
    num_value_dim  = X_num_train.shape[1]   # 7 (after dropping any constant or all-NaN cols)
    hidden_size    = 64
    static_hidden  = 32

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
        # 9.1 Training phase
        model.train()
        running_loss = 0.0

        train_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]", leave=False)
        for X_seq_batch, X_num_batch, X_cat_batch, y_batch in train_iter:
            X_seq_batch = X_seq_batch.to(device)  # (bs, 3, 10)
            X_num_batch = X_num_batch.to(device)  # (bs, num_value_dim)
            X_cat_batch = X_cat_batch.to(device)  # (bs, 4)
            y_batch     = y_batch.to(device)      # (bs,)

            optimizer.zero_grad()
            y_pred = model(X_seq_batch, X_num_batch, X_cat_batch)  # (bs,)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_seq_batch.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # 9.2 Validation phase
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

        # Save best model
        if epoch_val_loss < best_val_loss:
            print(f"[Info] Epoch {epoch} validation loss improved: {best_val_loss:.4f} → {epoch_val_loss:.4f}. Saving model.")
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_voyage_model.pth")

    print(f"\nTraining complete! Best validation loss = {best_val_loss:.4f}\n")

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
            X_seq_batch = X_seq_batch.to(device)  # (bs, 3, 10)
            X_num_batch = X_num_batch.to(device)  # (bs, num_value_dim)
            X_cat_batch = X_cat_batch.to(device)  # (bs, 4)
            y_batch     = y_batch.to(device)      # (bs,)

            y_pred = model(X_seq_batch, X_num_batch, X_cat_batch)  # (bs,)

            mse_loss_batch = mse_criterion(y_pred, y_batch)
            test_mse_loss += mse_loss_batch.item()

            mae_loss_batch = mae_criterion(y_pred, y_batch)
            test_mae_loss += mae_loss_batch.item()

            total_samples += y_batch.size(0)

    test_mse = test_mse_loss / total_samples
    test_mae = test_mae_loss / total_samples

    print(f"Test MSE Loss: {test_mse:.4f}")
    print(f"Test MAE Loss: {test_mae:.4f}")


if __name__ == '__main__':
    main()
