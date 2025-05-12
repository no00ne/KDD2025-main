import pandas as pd

csv_file_path = 'news/selected_news.csv'
df = pd.read_csv(csv_file_path, encoding='utf-8')
# 打印df前几行数据的timestamp列
print(df['timestamp'].head())