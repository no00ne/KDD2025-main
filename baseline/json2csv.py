import json
import pandas as pd
import os
import glob

# 設定輸入和輸出目錄
input_dir = 'ship_trajectories'
output_dir = 'ship_trajectories_csv'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 找到所有 .jsonl（實際上是完整的 JSON array）
json_files = glob.glob(os.path.join(input_dir, '*.jsonl'))

for json_file in json_files:
    print(f'Processing file: {json_file}')
    try:
        # 1) 把整個 JSON array 讀進來
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)   # raw_data 應該是一個 list

        if not isinstance(raw_data, list):
            raise ValueError(f"文件 {json_file} 不是 JSON array，而是 {type(raw_data)}")

        # 2) 用來暫存所有船舶的 Path DataFrame
        df_list = []

        for idx, ship_record in enumerate(raw_data):
            # 如果這條記錄不是 dict 或缺少 "Path"，就跳過
            if not isinstance(ship_record, dict) or "Path" not in ship_record:
                print(f'  【警告】 第 {idx} 個元素不是 dict 或缺少 "Path" 字段，跳過')
                continue

            path_list = ship_record["Path"]
            # 把這條船的 Path 轉成 DataFrame
            df_path = pd.DataFrame(path_list)

            # 檢查 "timestamp" 欄是否存在
            if 'timestamp' not in df_path.columns:
                print(f'  【警告】 ship_record[{idx}] 的 Path 裡缺少 timestamp，跳過')
                continue

            # 轉換 timestamp、排序
            df_path['timestamp'] = pd.to_datetime(df_path['timestamp'])
            df_path = df_path.sort_values('timestamp').reset_index(drop=True)

            # 只保留需要的列（如果某些列不在，就自動忽略）
            features = ['timestamp', 'latitude', 'longitude', 'speed', 'heading', 'course']
            existing = [f for f in features if f in df_path.columns]
            df_path = df_path[existing].copy()

            df_list.append(df_path)

        # 如果沒有任何有效的 Path，就跳過整個文件
        if len(df_list) == 0:
            print(f'  沒有可用的 Path，跳過 {json_file}')
            continue

        # 3) 把所有船舶的 Path 合併成一個大 DataFrame
        combined_df = pd.concat(df_list, ignore_index=True)

        # （可選）如果希望整個文件內按時間再做一次全局排序，可以取消下面這一行的註釋：
        # combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)

        # 生成輸出 CSV 名稱：把 .jsonl 換成 .csv
        base_name = os.path.basename(json_file).replace('.jsonl', '.csv')
        csv_path = os.path.join(output_dir, base_name)

        # 一次性寫出 CSV
        combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'  已成功生成 CSV：{base_name}')

    except Exception as e:
        print(f'處理文件 {json_file} 時發生錯誤: {str(e)}')

print('所有文件處理完成！')
