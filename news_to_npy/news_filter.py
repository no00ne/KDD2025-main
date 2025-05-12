import pandas as pd
import numpy as np
from datetime import datetime

# 读取CSV文件
try:
    # 尝试读取CSV文件
    df = pd.read_csv('news/news.csv')
    print(f"成功读取文件，总共有 {len(df)} 条记录")
except Exception as e:
    print(f"读取文件时出错: {e}")
    exit(1)

# 检查必要的列是否存在
required_columns = ['timestamp', 'content']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    print(f"CSV文件缺少必要的列: {', '.join(missing_columns)}")
    exit(1)

# 筛选步骤1: 删除timestamp或content为空的行
initial_count = len(df)
df = df.dropna(subset=['timestamp', 'content'])
print(f"删除了 {initial_count - len(df)} 条timestamp或content为空的记录")

# 筛选步骤2: 转换timestamp为datetime格式
try:
    # 使用ISO8601格式，这应该能处理带时区的时间戳
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
    print("成功将timestamp列转换为datetime格式")
except Exception as e:
    print(f"使用ISO8601格式转换timestamp时出错: {e}")
    try:
        # 尝试混合格式解析
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed')
        print("使用混合格式成功转换timestamp")
    except Exception as e:
        print(f"使用混合格式转换timestamp时出错: {e}")
        try:
            # 最后尝试不指定格式，让pandas自动推断
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # 检查是否有无法解析而变成NaT的行
            nat_count = df['timestamp'].isna().sum()
            if nat_count > 0:
                print(f"警告: {nat_count} 条记录的timestamp无法解析，这些记录将被丢弃")
                df = df.dropna(subset=['timestamp'])
            print("使用自动推断成功转换timestamp")
        except Exception as e:
            print(f"所有尝试都失败: {e}")
            print("无法解析timestamp列，请检查数据格式")
            exit(1)

# 筛选步骤3: 选择2021年1月1日到2021年6月30日的数据
start_date = '2021-01-01'
end_date = '2021-06-30'
mask = (df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)
filtered_df = df.loc[mask]

print(f"筛选出 {len(filtered_df)} 条在2021年1月1日至2021年6月30日之间的记录")

# 筛选步骤4: 按照timestamp进行降序排序
filtered_df = filtered_df.sort_values(by='timestamp', ascending=True)

# 保存结果到新的CSV文件
try:
    filtered_df.to_csv('../news/selected_news.csv', index=False)
    print(f"成功将结果保存到selected_news.csv文件中")
except Exception as e:
    print(f"保存文件时出错: {e}")