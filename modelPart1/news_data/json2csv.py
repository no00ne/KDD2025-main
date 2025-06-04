import json
import pandas as pd

# 使用明確的相對路徑
with open("news_data/unpredictable.json", "r") as f:
    data = json.load(f)

# 展平嵌套结构
flattened = []
for item in data:
    row = {
        "event_time": item["event_time"],
        "news_id": item["news_id"],
    }

    # 展开 scores
    for idx, score in enumerate(item["scores"]):
        row[f"score_{idx}"] = score

    # 展开 bounding_box
    bbox = item["bounding_box"]
    row.update({
        "north": bbox["north"],
        "south": bbox["south"],
        "east": bbox["east"],
        "west": bbox["west"],
    })

    flattened.append(row)

# 保存为 CSV
df = pd.DataFrame(flattened)
# 将新闻数据按照 event_time 排序
df = df.sort_values(by="event_time")
df.to_csv("news_data/unpredictable.csv", index=False)
