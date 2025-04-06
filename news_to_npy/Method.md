### Abstract
    原论文首先将新闻文本分为预测和已然两种，声明为post_treats和pre_treats。
    接着根据发生地点和时间（原文中日本被分为几百个行政区，时间按小时划分）以及大模型对于事件进行10维度的打分
    并最终按照时间和地点进行聚合，得到一个600*490*10的矩阵

### API
    define a file named config.py and add your API key

### Prompt
    System:
        You are an AI assistant that notifies affected users and makes suggestions to change their mobility based on news text. 
    Prompt1 Public Events:
        First, identify the most influential events in the news content, including scheduled and unpredictable events
    Prompt2 Time Information:
        Next, estimate the exact time of the most important event mentioned in the news in the following JSON format: “event time”: “yyyy-mm-dd hh:mm:ss”. If the exact time is unknown, use the news release time. Provide only the JSON string without any additional text or explanations.
    Prompt3 3W1H about Human Mobility:
        Based on the news text and our chat, evaluate the 3W1H related to human mobility.
            - Where: Where should people move if necessary?
            - Who: What kind of people will be affected?
            - When: When did the event happen?
            - How: How should people move if necessary
    Prompt4 Predictability: 
        Is the content in the news more like an unpredictable event, such as an earthquake? Your answer can only be “Yes” or “No”.
    Prompt5 Human Intentions:
        Background: Most news related to economic, politics, culture, and history issues usually have no effect on human mobility because they do not relate to people’s daily life.
            – Slight disasters, such as light earthquakes and tsunamis, or some local events (like political events) may also have no effect on human mobility in Japan.
            – Only events that happened close to the release time (within several hours) may have an influence on human mobility.
        Task: Score the news text based on the following aspects (0–100, where a higher number means higher agreement):

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

        Expected response: A list of 10 numbers between 0 and 100.