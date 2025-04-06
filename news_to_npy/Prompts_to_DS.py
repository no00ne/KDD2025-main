import http.client
import json
from config import API_KEY

def analyze_news(news_content):
    # DeepSeek API配置
    HOST = "api.deepseek.com"
    ENDPOINT = "/chat/completions"

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    # 通用system message
    system_prompt = {
        "role": "system",
        "content": "You are an AI assistant that notifies affected users and makes suggestions to change their mobility based on news text."
    }

    # Prompts
    prompts = [
        "First, identify the most influential events in the news content, including scheduled and unpredictable events",

        "Next, estimate the exact time of the most important event mentioned in the news in the following JSON format: “event time”: “yyyy-mm-dd hh:mm:ss”. If the exact time is unknown, use the news release time. Provide only the JSON string without any additional text or explanations.",

        "Based on the news text and our chat, evaluate the 3W1H related to human mobility.\n- Where: Where should people move if necessary?\n- Who: What kind of people will be affected?\n- When: When did the event happen?\n- How: How should people move if necessary",

        "Is the content in the news more like an unpredictable event, such as an earthquake? Your answer can only be “Yes” or “No”.",

        """Score the news text based on the following aspects (0–100, where a higher number means higher agreement):
            Q1. To what extent do the events described in the news make people leave the area because they are dangerous?
            Q2. To what extent do the events described in the news make people stay in the area because it is better not to move?
            Q3. To what extent do the events described in the news make people visit the area because they are interesting events?
            Q4. To what extent do the events described in the news make people keep their daily routine as these events are not important to daily life?
            Q5. To what extent do the events described in the news lead to interruption of economic activities, such as business closures or work stoppages?
            Q6. To what extent do the events described in the news affect transportation conditions, such as traffic congestion or road closures?
            Q7. To what extent do the events described in the news impact public health and safety, leading to decisions to leave or avoid certain areas?
            Q8. To what extent do the events described in the news involve government or official instructions that influence people’s movements?
            Q9. To what extent do the events described in the news affect the availability of public services, such as school closures or interruptions in medical services?
            Q10. To what extent do the events described in the news last a long time (like one day)?
            \nExpected response: A list of 10 numbers between 0 and 100."""
    ]

    messages = [system_prompt]
    conn = http.client.HTTPSConnection(HOST)

    # 构造对话轮次
    for i, prompt in enumerate(prompts):
        user_msg = {
            "role": "user",
            "content": f"{prompt}\n\nThe news is as follows:\n{news_content}"
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

        except Exception as e:
            print(f"\n--- Prompt{i+1} Response (Error) ---")
            print(e)
            break

    conn.close()

if __name__ == "__main__":
    news = "A 6.1-magnitude earthquake has struck Japan's Osaka region, killing at least three people and injuring dozens more.The quake struck at 7:58 a.m. The epicentre was located at 34.8 degrees north latitude and 135.6 degrees east longitude at a depth of 10 kilometres."
    analyze_news(news)