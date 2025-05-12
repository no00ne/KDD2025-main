import http.client
import re
from datetime import datetime
from config import config as config
import pandas as pd
import json
import time
import traceback

from tqdm import tqdm

import os
from openai import OpenAI

import threading
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def aly_analyze_news(news_object):

    # OpenAI兼容API配置
    client = OpenAI(
        api_key=config.ALY_API_KEY,
        base_url=config.ALY_base_url
    )
    model = "deepseek-r1"

    precise_content = config.precise_content

    # 通用system message
    system_prompt = {
        "role": "system",
        "content": precise_content,
    }

    # Prompts
    prompts = config.prompts

    messages = [system_prompt]
    responses = {}

    # 构造对话轮次
    for i, prompt in enumerate(prompts):
        user_msg = {
            "role": "user",
            "content": f"{prompt}\n\nThe news is as follows:\n{news_object}"
        }
        messages.append(user_msg)

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1,
                max_tokens=4096
            )

            assistant_reply = completion.choices[0].message.content
            messages.append({
                "role": "assistant",
                "content": assistant_reply
            })

            print(f"\n--- Prompt{i + 1} Response ---")
            print(assistant_reply.strip())
            responses[i] = assistant_reply.strip()

        except Exception as e:
            print(f"\n--- Prompt{i + 1} Response (Error) ---")
            print(e)
            break

    return responses

def llm_analyze_news(news_object):

    # DeepSeek API配置
    HOST = "api.deepseek.com"
    ENDPOINT = "/chat/completions"
    API_KEY = config.DS_KEY
    model = "deepseek-chat" # deepseek-v3

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    precise_content = config.precise_content

    # 通用system message
    system_prompt = {
        "role": "system",
        "content": precise_content,

    }

    # Prompts
    prompts = config.prompts

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
            "model": model,
            "temperature": 1,
            "max_tokens": 4096
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

            # print(f"\n--- Prompt{i+1} Response ---")
            # print(assistant_reply.strip())
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


def news_classification(evaluations):
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


def process_single_news(row_data, source="ds", max_retries=3):
    """
    处理单条新闻的函数，适用于并行处理

    参数:
        row_data: 包含(index, row)的元组，row是pandas的Series对象
        source: 使用的LLM服务源，"ds"或"aly"
        max_retries: 最大重试次数

    返回:
        (response, evaluation, success_flag, news_id): 包含LLM响应、评估结果、处理成功标志和新闻ID
    """
    index, row = row_data
    news_id = row.get('id', index)  # 如果没有id列，使用索引作为id
    content = row.get('content', '').strip()

    if not content or pd.isna(content):
        print(f"警告: 第{index}行新闻内容为空，跳过")
        return None, None, False, news_id

    timestamp = row.get('timestamp', '')
    content = "News Release Date: " + str(timestamp) + "\n" + content

    # 使用线程安全的方式打印
    print_lock = threading.Lock()
    with print_lock:
        print(f"\n处理第{index}行新闻 (ID: {news_id})")

    # 尝试处理，最多重试max_retries次
    for attempt in range(max_retries):
        try:
            # 分析新闻
            if source == "ds":
                response = llm_analyze_news(content)
            elif source == "aly":
                response = aly_analyze_news(content)

            # 转换为评估结果
            evaluation = translate_output_to_news_evaluation(response)

            # 添加新闻ID
            response['news_id'] = news_id
            evaluation['news_id'] = news_id

            with print_lock:
                print(f"成功处理第{index}行新闻")

            return response, evaluation, True, news_id

        except Exception as e:
            if attempt < max_retries - 1:
                with print_lock:
                    print(f"处理第{index}行新闻时出错（尝试 {attempt + 1}/{max_retries}）: {str(e)}")
                    print("正在重试...")
                time.sleep(2)  # 等待短暂时间再重试
            else:
                with print_lock:
                    print(f"处理第{index}行新闻失败（尝试 {max_retries}/{max_retries}）:")
                    print(traceback.format_exc())
                    print("跳过此条新闻")

    return None, None, False, news_id


def save_results(all_responses, all_evaluations, response_file, evaluation_file):
    """使用线程锁保存结果到文件"""
    file_lock = threading.Lock()
    with file_lock:
        with open(response_file, "w", encoding="utf-8") as file:
            json.dump(all_responses, file, ensure_ascii=False, indent=4)

        with open(evaluation_file, "w", encoding="utf-8") as file:
            json.dump(all_evaluations, file, ensure_ascii=False, indent=4)


def process_selected_news(input_file,
                          response_file="responses.json",
                          evaluation_file="evaluations.json",
                          source="ds",
                          max_retries=3,
                          num_workers=None):
    """
    从CSV文件中读取新闻内容，使用线程池并行方式调用LLM分析每条新闻

    参数:
        input_file (str): 输入CSV文件路径
        response_file (str): LLM响应保存文件
        evaluation_file (str): 评估结果保存文件
        source (str): 使用的LLM服务源，"ds"或"aly"
        max_retries (int): 最大重试次数
        num_workers (int): 线程数，如果为None则使用较大的默认值
    """
    if num_workers is None:
        # 由于是IO绑定任务，线程数可以设置得较高
        num_workers = min(32, os.cpu_count() * 4)  # 限制最大线程数为32

    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)
        print(f"成功读取{input_file}，共有{len(df)}条新闻")
        print(f"将使用 {num_workers} 个线程并行处理")

        # 准备存储结果的列表 (线程共享内存，可以直接使用普通列表)
        all_responses = []
        all_evaluations = []
        results_lock = threading.Lock()

        # 创建一个partial函数，固定source和max_retries参数
        process_func = partial(process_single_news, source=source, max_retries=max_retries)

        # 使用线程池
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # 提交所有任务
            futures = {executor.submit(process_func, (i, row)): i for i, row in df.iterrows()}

            # 使用tqdm显示进度
            with tqdm(total=len(futures), desc="处理新闻") as pbar:
                # 处理完成的任务
                saved_counter = 0
                for future in as_completed(futures):
                    response, evaluation, success, news_id = future.result()
                    if success:
                        with results_lock:
                            all_responses.append(response)
                            all_evaluations.append(evaluation)
                            saved_counter += 1

                            # 每处理10条新闻保存一次结果，避免中断损失
                            if saved_counter % 10 == 0:
                                save_results(all_responses, all_evaluations,
                                             response_file, evaluation_file)
                    pbar.update(1)

        # 最终保存结果
        save_results(all_responses, all_evaluations, response_file, evaluation_file)

        print(f"\n处理完成! 成功处理 {len(all_responses)} 条新闻")
        return all_responses, all_evaluations

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        print(traceback.format_exc())
        return [], []


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "news", "selected_news.csv")

    try:
        # 处理CSV文件中的所有新闻, ds
        process_selected_news(
            input_file=input_file,
            response_file="responses-v3.json",
            evaluation_file="evaluations-v3.json",
            source="ds",  # "ds" or "aly"
            max_retries=3,
            num_workers=25  # 对于网络IO密集型任务，可使用更多线程
        )

        # 处理CSV文件中的所有新闻, aly
        process_selected_news(
            input_file=input_file,
            response_file="result/responses-R1.json",
            evaluation_file="result/evaluations-R1.json",
            source="aly",
            max_retries=3,
            num_workers=25
        )
    except KeyboardInterrupt:
        print("检测到用户中断，正在优雅地停止处理...")
        # 线程池会处理中断

    # example_precess()


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

    # 记录运行时间
    start_time = time.time()

    response = aly_analyze_news(news)
    with open("result/response.json", "w", encoding="utf-8") as file:
        json.dump(response, file, ensure_ascii=False, indent=4)
    evaluation = translate_output_to_news_evaluation(response)
    print(evaluation)
    with open("result/evaluation.json", "w", encoding="utf-8") as file:
        json.dump(evaluation, file, ensure_ascii=False, indent=4)

    # 运行时间
    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f}秒")
