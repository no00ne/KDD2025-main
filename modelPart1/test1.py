import time
import torch
import numpy as np
import pandas as pd

UN_PRED_NEWS = pd.read_csv("news_data/unpredictable.csv")
PRED_NEWS    = pd.read_csv("news_data/predictable.csv")

# 预先提取常用列为 numpy 数组，加快后续广播计算
up_arr  = UN_PRED_NEWS[["event_time","north","south","east","west",
                        *[f"score_{i}" for i in range(6)]]].values
pr_arr  = PRED_NEWS [ ["event_time","north","south","east","west",
                        *[f"score_{i}" for i in range(6)]]].values

up_times = up_arr[:,0]
pr_times = pr_arr[:,0]
# ------------------------------------

def get_node_related_news_tensor(nodes, max_num=10, projection=False):
    """
    Faster implementation for large-scale scenarios.
    Returns: torch.Tensor shape (num_nodes, max_num, 7)
    """
    num_nodes = len(nodes)
    out = torch.zeros((num_nodes, max_num, 7), dtype=torch.float32)

    for idx, node in enumerate(nodes):
        t, lat, lon = node["timestamp"], node["latitude"], node["longitude"]

        # --- 时间切片（利用 searchsorted） ---
        up_cut = np.searchsorted(up_times, t, side="right")
        pr_cut = np.searchsorted(pr_times, t, side="left")
        up_slice = up_arr[:up_cut]
        pr_slice = pr_arr[pr_cut:]

        # --- 合并后做空间过滤（numpy 广播） ---
        slice_arr = np.vstack((up_slice, pr_slice))
        if slice_arr.size == 0:
            continue

        lat_ok = (slice_arr[:,1] >= lat) & (slice_arr[:,2] <= lat)
        lon_ok = (slice_arr[:,3] >= lon) & (slice_arr[:,4] <= lon)
        hits   = slice_arr[lat_ok & lon_ok]

        if hits.shape[0]:
            # 取前 max_num 条
            hits = hits[:max_num]

            # event_time 或差值
            times = hits[:,0:1] - t if projection else hits[:,0:1]
            scores = hits[:,5:11]     # 6 列 score
            tensor = torch.tensor(np.hstack((times, scores)), dtype=torch.float32)

            out[idx, :tensor.shape[0], :] = tensor

    return out

# --------- 示例节点生成 ---------
def sample_nodes():
    """
    Return a list of 3 dummy nodes for quick testing.
    Each node has timestamp, latitude, longitude fields.
    """
    now = int(time.time())          # current Unix timestamp
    return [
        {"timestamp": now - 3600, "latitude": 32.0, "longitude": 125.0},
        {"timestamp": now - 1800, "latitude": 33.5, "longitude": 130.0},
        {"timestamp": now +  600, "latitude": 31.0, "longitude": 120.0},
    ]

# --------- 测试函数 ---------
def test_get_node_related_news_tensor():
    nodes = sample_nodes()
    tensor = get_node_related_news_tensor(nodes, max_num=5, projection=True)

    # Basic sanity checks
    print("Tensor shape :", tensor.shape)   # should be (3, 5, 7)
    print("Dtype        :", tensor.dtype)

    # Show first node's first 2 news rows (if any)
    print("First node, first two rows :\n", tensor[0, :2])

    # Confirm padding rows are all-zero
    pad_row = tensor[0, -1]   # last row
    if torch.allclose(pad_row, torch.zeros(7)):
        print("Padding row is all zeros ✔")
    else:
        print("Padding row not zero ✘")

if __name__ == "__main__":
    test_get_node_related_news_tensor()
