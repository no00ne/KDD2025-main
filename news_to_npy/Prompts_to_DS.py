import http.client
import json
import re
from datetime import datetime

from config import API_KEY


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
            Q1. To what extent do the events described in the news cause vessels to reroute due to safety concerns?
            Q2. To what extent do the events described in the news cause vessels to stay in port or delay departure?
            Q3. To what extent do the events described in the news increase traffic in certain shipping lanes or ports?
            Q4. To what extent do the events described in the news have minimal impact on normal shipping schedules?
            Q5. To what extent do the events described in the news affect cargo operations, such as loading/unloading delays?
            Q6. To what extent do the events described in the news impact navigational conditions, such as channel restrictions or closures?
            Q7. To what extent do the events described in the news create safety hazards that might cause vessels to slow down or proceed with caution?
            Q8. To what extent do the events described in the news involve maritime authorities issuing instructions that influence vessel movements?
            Q9. To what extent do the events described in the news affect port services, such as pilotage, tugs, or terminal operations?
            Q10. To what extent do the events described in the news represent a long-duration impact (more than 24 hours) on shipping?
            \nExpected response: A list[] of 10 numbers between 0 and 100.
        """,

        # 让llm预测可能的影响范围（以经纬度的形式），最终从该类椭圆中提取被包含在内的格子信息
        """
        Based on the news content, estimate the potential geographical area affected by the described event. 
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
    print(evaluations)
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


def news_decoder(classified_news: []):
    """
        对于包含时间，影响范围，评分列表的新闻，进行进一步的解码，使得能顺利输入模型
        论文中的思路：
            1. 首先将时间划分为n个片段，时间段的长度为 $(?)
            2. 接着将影响范围分解为m个范围
            3. 最后将数据转化为 n*m*10 的矩阵，n为时间段的个数，m为影响范围的个数，10为评分列表的长度

        parameter：
            一个列表classified_news：已经分类并包含必要信息的llm评估结果

        return：
            一个npy文件，包含解析后的结果
    """

if __name__ == "__main__":
    news = """
    !Houthi claim attack on vessel Abliani and other vessels.

    Reuters media reported that Houthis claimed to have attacked a U.S. aircraft carrier, a U.S. destroyer, and three vessels. Houthis spokesperson stated on Saturday, June 01, that they have targeted a U.S.aircraft carrier, Eisenhower, a U.S.destroyer and three vessels, namely ABLIANI, MAINA, and AL ORAIQ, sailing in the Red Sea and the Indian Ocean. 

    The spokesperson stated that they targeted the U.S. aircraft Carrier Eisenhower at the north of the Red Sea. The U.S. aircraft carrier Eisenhower was attacked by several missiles and drones. Subsequently, the Houthi spokesperson claimed to attack a U.S. destroyer and crude oil tanker, ABLIANI, sailing in the Red Sea. ABLIANI is a 10,9999 dwt crude oil tanker sailing under the flag of Malta. On June 01, the vessel departed from JAZAN ECONOMIC CITY Port, Saudi Arabia and is scheduled to arrive at SUEZ CANAL, Egypt on June 04.

    Following this, the spokesperson of Houthi also claimed to attack twice on the vessel MAINA. The bulk carrier MAINA was targeted in the Red Sea and then in the Arabian Sea. MAINA is a bulk carrier registered in Malta. The bulk carrier departed from Port UST-LUNGA, Russia on May 07 en route to Port KRISHNAPATNAM, India.  He further added that another vessel, AL ORAIQ, was also targeted in the Indian Ocean. AL ORAIQ is an LNG Carrier with a capacity of 2,05,994 cubic meters LNG. The Marshall Island flagged vessel departed from RAS LAFFAN, Qatar on May 27, destined for Port CHIOGGIA, Italy. 

    On June 02, The U.S. Central Command (USCENTCOM) announced in its press release that on June 01, USCENTCOM forces destroyed one Houthi uncrewed aerial system(UAS) in the southern Red Sea. The USCENTCOM forces also identified two other UAS that crashed into the Red Sea. The U.S., coalition, or other commercial ships reported no casualties.

    Moreover, the USCENTCOM forces have also destroyed two Houthi anti-ship ballistic missiles (ASBM) in the southern Red Sea. The ASBM was launched in the direction of USS Gravely but was destroyed by USCENTCOM.   

    Houthis have claimed these attacks on vessels and U.S. carriers and destroyers after USCENTCOM and U.K. armed forces carried out strikes against 13 Houthis terrorist-controlled areas in Yemen on May 30.  

    """

    response = llm_analyze_news(news)
    with open("my_dict.json", "w", encoding="utf-8") as file:
        json.dump(response, file, ensure_ascii=False, indent=4)
    evaluation = translate_output_to_news_evaluation(response)
    print(evaluation)
