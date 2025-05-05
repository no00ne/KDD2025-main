import http.client
import re
from datetime import datetime
from config import API_KEY
import pandas as pd
import json
import time
import traceback
from tqdm import tqdm


def generate_news_object(news_df):
    """
    将新闻数据 DataFrame 转换为格式化字符串列表，每条为：
    title: ...
    release timestamp: ...
    content: ...

    parameter：
        news_df (pd.DataFrame): 包含 'title'、'timestamp' 和 'content' 列的 DataFrame

    return：
        List[str]: 每条新闻的格式化字符串
    """
    news_objects = []
    for _, row in news_df.iterrows():
        formatted = f"**title**: {row['title']}\n**release timestamp**: {row['timestamp']}\n**content**: {row['content']}"
        news_objects.append(formatted)
    return news_objects


def llm_analyze_news(news_object):
    # news_object: **title**: ..., **release timestamp**: ..., **content**: ...,

    # DeepSeek API配置
    HOST = "api.deepseek.com"
    ENDPOINT = "/chat/completions"

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    # TODO: compare these two
    precise_content = (
        "You are a concise and objective AI assistant. "
        "Your task is to directly respond to the user's instructions based solely on the given news content. "
        "Do not provide specific analysis, suggestions, especially be careful not to use Markdown format (like **). "
        "Only respond to what is asked. Keep the output plain, precise and well thought out."
    )
    # 原论文的prompt content
    origin_content = "You are an AI assistant that notifies affected users and makes suggestions to change their mobility based on news text."

    # 通用system message
    system_prompt = {
        "role": "system",
        "content": precise_content,

    }

    # Prompts
    prompts = [
        """
        First, identify the most influential maritime events in the news content, 
        including scheduled operations and unpredictable incidents that may affect shipping schedules.
        """,

        # 由于csv中的time为发布时间，因此需要从新闻原文中获取实际开始造成影响的时间
        # 新闻中存在没有显式说明时间的问题，考虑到新闻的时效性，可以使用新闻发布时间替代
        # prompt中使用 exact time of the most important event mentioned in the news 来避免输出多个时间
        """
        Next, estimate the exact time of the most important event mentioned in the news in the following JSON format: “event time”: “yyyy-mm-dd hh:mm:ss”. 
        If the exact time is unknown, use the news release time. Provide only the JSON string without any additional text or explanations.
        """,

        """
        Based on the news text and our chat, evaluate the key factors related to shipping ETA.\n
        - Where: Which shipping routes or ports are affected?\n
        - What: What type of vessels will be affected?\n
        - When: When did/will the event happen?\n
        - How: How will shipping schedules be impacted?
        """,

        # 让llm判断这是否是一个预测性质的新闻
        "Is the content in the news more like an unpredictable event, such as an earthquake? Your answer can only be “Yes” or “No”.",

        # 让llm给出事件影响因子
        """
        Score the news text based on the following aspects (0–100, where a higher number means higher agreement):
            Q1. To what extent do the events described in the news cause vessels to reroute?
            Q2. To what extent do the events described in the news cause vessels to stay in port or delay departure?
            Q3. To what extent do the events described in the news increase traffic in certain shipping lanes or ports?
            Q4. To what extent do the events described in the news impact navigational conditions, such as channel restrictions or closures?
            Q5. To what extent do the events described in the news create safety hazards that might cause vessels to slow down or proceed with caution?
            Q6. To what extent do the events described in the news represent a long-duration impact (more than 24 hours) on shipping?
            \nExpected response: A list[] of  numbers between 0 and 100.
        """,

        # 让llm预测可能的影响范围（以经纬度的形式），最终从该类椭圆中提取被包含在内的格子信息
        """
        Based on the news content and the score list you just gave, estimate the potential geographical area affected by the described event. The more precise the better.
        If the range of influence is too large, please consider carefully whether the range is truly reliable.
        Return a bounding box that covers the impacted region using the following JSON format:

        {
          "bounding box": {
            "north": <latitude>,
            "south": <latitude>,
            "east": <longitude>,
            "west": <longitude>
          }
        }

        If the affected area cannot be determined from the news, return null. Provide only the JSON object without additional explanation.
        """,
    ]

    messages = [system_prompt]
    conn = http.client.HTTPSConnection(HOST)

    responses = {}

    # 构造对话轮次
    for i, prompt in enumerate(prompts):
        user_msg = {
            "role": "user",
            "content": f"{prompt}\n\nThe news is as follows:\n{news_object}"
        }
        messages.append(user_msg)

        payload = json.dumps({
            "messages": messages,
            "model": "deepseek-chat",
            "temperature": 1,
            "max_tokens": 2048
        })

        conn.request("POST", ENDPOINT, payload, headers)
        res = conn.getresponse()
        data = res.read()

        try:
            response_json = json.loads(data)
            assistant_reply = response_json["choices"][0]["message"]["content"]
            messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

            print(f"\n--- Prompt{i+1} Response ---")
            print(assistant_reply.strip())
            responses[i] = assistant_reply.strip()

        except Exception as e:
            print(f"\n--- Prompt{i+1} Response (Error) ---")
            print(e)
            break

    conn.close()
    return responses


def translate_output_to_news_evaluation(llm_output):
    """
        将 llm_output 提取并存储为 dict{时间，是否为预测的新闻，评分列表，影响范围}

        parameter：
            llm根据新闻给出的输出

        return：
            dict{时间(timestamp)，是否为预测的新闻(bool)，评分列表([])，影响范围({})}，其中时间和影响范围可能为None
    """
    # 提取并清洗时间（字段 1）
    time_str_raw = llm_output.get(1, "")
    # 去除```json```标记并加载为 dict
    time_json_clean = re.sub(r"```json|```", "", time_str_raw).strip()
    event_time_str = json.loads(time_json_clean).get("event time", None)
    event_time = None
    if event_time_str:
        dt_obj = datetime.strptime(event_time_str, "%Y-%m-%d %H:%M:%S")
        event_time = int(dt_obj.timestamp())

    # 提取是否为不可预测事件（字段 3）
    unpredictable_str = llm_output.get(3, "").strip().lower()
    is_unpredictable = unpredictable_str == "Yes"

    # 提取评分列表（字段 4）
    scores_str = llm_output.get(4, "")
    scores = json.loads(scores_str) if isinstance(scores_str, str) else scores_str

    # 提取影响范围（字段 5）
    bbox_raw = llm_output.get(5, "")
    bbox_json_clean = re.sub(r"```json|```", "", bbox_raw).strip()
    bounding_box = json.loads(bbox_json_clean).get("bounding box", None)

    return {
        "event_time": event_time,
        "is_unpredictable": is_unpredictable,
        "scores": scores,
        "bounding_box": bounding_box
    }


def news_classification(evaluations: []):
    """
        根据评估结果列表中的元素 dict{时间，是否为预测的新闻，评分列表，影响范围}，进行进一步的解码，使得能顺利输入模型
        首先根据 dict 中是否存在 None ，如果是，则抛弃。
        接着根据 is_unpredictable ,如果是，则将其归入 prev，如果不是，则将其归入 post。归入的同时将 is_unpredictable 删去。

        parameter：
            一个列表，列表内是llm根据新闻给出的输出（以字典的形式）

        return：
            1. 可预测新闻组成的列表dict{时间，评分列表，影响范围}
            2. 已经发生的新闻组成的列表dict{时间，评分列表，影响范围}
    """
    # 过滤掉包含 None 的评估结果
    filtered_evaluations = [content for content in evaluations if all(value is not None for value in content.values())]

    # 将评估结果分为可预测和已发生的新闻
    prev_news = []
    post_news = []

    for content in filtered_evaluations:
        if content["is_unpredictable"]:
            prev_news.append({k: v for k, v in content.items() if k != "is_unpredictable"})
        else:
            post_news.append({k: v for k, v in content.items() if k != "is_unpredictable"})

    return prev_news, post_news


def process_selected_news(input_file="..\\news\\selected_news.csv",
                          response_file="responses.json",
                          evaluation_file="evaluations.json",
                          max_retries=3):
    """
    从CSV文件中读取新闻内容，使用LLM分析并评估每条新闻

    参数:
        input_file (str): 输入CSV文件路径
        response_file (str): LLM响应保存文件
        evaluation_file (str): 评估结果保存文件
        max_retries (int): 最大重试次数
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        print(f"成功读取{input_file}，共有{len(df)}条新闻")

        # 准备存储结果的列表
        all_responses = []
        all_evaluations = []

        # 逐行处理新闻
        for index, row in tqdm(df.iterrows(), total=len(df), desc="处理新闻"):
            news_id = row.get('id', index)  # 如果没有id列，使用索引作为id
            content = row.get('content', '').strip()

            if not content or pd.isna(content):
                print(f"警告: 第{index}行新闻内容为空，跳过")
                continue

            timestamp = row.get('timestamp', '')
            content = "News Release Date: " + str(timestamp) + "\n" + content

            print(f"\n处理第{index}行新闻 (ID: {news_id})")

            # 尝试处理，最多重试max_retries次
            for attempt in range(max_retries):
                try:
                    # 分析新闻
                    response = llm_analyze_news(content)

                    # 转换为评估结果
                    evaluation = translate_output_to_news_evaluation(response)

                    # 添加新闻ID
                    response['news_id'] = news_id
                    evaluation['news_id'] = news_id

                    # 保存到列表
                    all_responses.append(response)
                    all_evaluations.append(evaluation)

                    # 实时保存结果到文件
                    with open(response_file, "w", encoding="utf-8") as file:
                        # 由于每次都是整个[]的写入，因此可以使用 'w'
                        json.dump(all_responses, file, ensure_ascii=False, indent=4)

                    with open(evaluation_file, "w", encoding="utf-8") as file:
                        json.dump(all_evaluations, file, ensure_ascii=False, indent=4)

                    print(f"成功处理第{index}行新闻")
                    break  # 处理成功，跳出重试循环

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"处理第{index}行新闻时出错（尝试 {attempt + 1}/{max_retries}）: {str(e)}")
                        print("正在重试...")
                        time.sleep(2)  # 等待短暂时间再重试
                    else:
                        print(f"处理第{index}行新闻失败（尝试 {max_retries}/{max_retries}）:")
                        print(traceback.format_exc())
                        print("跳过此条新闻")

        print(f"\n处理完成! 成功处理 {len(all_responses)} 条新闻")
        return all_responses, all_evaluations

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        print(traceback.format_exc())
        return [], []


def example_precess():

    # 预测类新闻
    news = """
    Possible Escalation Expected as Houthis Claim Multiple Maritime Attacks in Red Sea and Indian Ocean

June 2, 2024 – Red Sea / Indian Ocean

Tensions in the Red Sea and surrounding waters may escalate following a series of attacks claimed by Yemen’s Houthi forces. On June 1, a Houthi spokesperson announced that the group had launched coordinated strikes against multiple U.S. naval assets and commercial vessels operating in the region.

According to Houthi statements, targets included the U.S. aircraft carrier Eisenhower, a U.S. Navy destroyer, and three commercial vessels: ABLIANI, MAINA, and AL ORAIQ. The attacks reportedly spanned across the Red Sea, Arabian Sea, and the Indian Ocean.

The Houthis claimed to have launched multiple missiles and drones targeting the Eisenhower north of the Red Sea. They also stated that the destroyer and the crude oil tanker ABLIANI, sailing under the Maltese flag, were targeted in the same region. The ABLIANI departed from Jazan Economic City Port in Saudi Arabia on June 1, en route to the Suez Canal, Egypt, with an estimated arrival date of June 4.

Further, the bulk carrier MAINA, also Malta-registered, was allegedly attacked twice — first in the Red Sea and subsequently in the Arabian Sea. The vessel had departed from Ust-Luga, Russia, on May 7 and was bound for Krishnapatnam, India. Another vessel, the LNG carrier AL ORAIQ, was reportedly attacked in the Indian Ocean. The Marshall Islands-flagged tanker departed Ras Laffan, Qatar, on May 27, heading to Chioggia, Italy.

On June 2, the U.S. Central Command (USCENTCOM) confirmed that on June 1, its forces destroyed one Houthi uncrewed aerial system (UAS) in the southern Red Sea. Additionally, two other UAS crashed into the sea, and no damage or casualties were reported by U.S., coalition, or commercial ships.

USCENTCOM also reported the interception and destruction of two Houthi-launched anti-ship ballistic missiles (ASBM) targeting the USS Gravely. These defensive actions come just two days after joint U.S. and U.K. forces conducted airstrikes on 13 Houthi-controlled targets in Yemen on May 30.
    """

    response = llm_analyze_news(news)
    with open("response.json", "w", encoding="utf-8") as file:
        json.dump(response, file, ensure_ascii=False, indent=4)
    evaluation = translate_output_to_news_evaluation(response)
    print(evaluation)
    with open("evaluation.json", "w", encoding="utf-8") as file:
        json.dump(evaluation, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':

    # 处理CSV文件中的所有新闻
    process_selected_news()

    # example_precess()
