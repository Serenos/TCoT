

import os
from openai import OpenAI


def call_zhipu():
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key="23211c113cb69c30b1860086d69c4709.ywrx7b3ND45ikedj") # Fill in your own APIKey
    prompt_example = 'str'
    completion = client.chat.completions.create(
        model="glm-4-plus",  # Fill in the model code you need to call
        messages=[
            {"role": "system", "content": "You are an assistant who loves to answer various questions, your task is to provide users with professional, accurate, and insightful advice."},
            {"role": "user", "content": f"{prompt_example}"}
        ],
    )
    with open('/home/lixiang/codebase/embodied-CoT/scripts/generate_embodied_data/prompt_answer_zhipu.txt', 'w') as f:
        f.write(completion.choices[0].message.content)

def call_qianwen(prompt, episode_id, prompt_pth, answer_pth):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
        api_key="sk-80c201fc880143b992b301629fc93d7a", 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    print('calling qianwen-vl-plus ...')
    completion = client.chat.completions.create(
        model="qwen-plus-0806", # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': f'{prompt}'}],
        temperature=0,
        top_p=0
        )
        
    #print(completion.model_dump_json())
    print(prompt_pth)
    with open(prompt_pth, 'w') as f:
        f.write(prompt)
    print('prompt file save to:', prompt_pth)

    with open(answer_pth, 'w') as f:
        f.write(completion.choices[0].message.content)
    print('answer file save to:', answer_pth)


def call_gemmi():
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted
    import time

    class Gemini:
        def __init__(self):
            api_key = "AIzaSyDtoThaSZAHAmFNeocDcQbmuu_iOiWJCFA"
            genai.configure(api_key=api_key)

            self.model = genai.GenerativeModel("gemini-1.5-flash")

        def safe_call(self, f):
            while True:
                try:
                    res = f()
                    return res
                except ResourceExhausted:
                    time.sleep(5)

        def generate(self, prompt):
            chat = self.safe_call(lambda: self.model.start_chat(history=[]))
            response = self.safe_call(lambda: chat.send_message(prompt).text)

            for i in range(8):
                if "FINISHED" in response:
                    print(f"n_retries: {i}")
                    return response

                response = response + self.safe_call(lambda: chat.send_message("Truncated, please continue.").text)

            print(f"n_retries: {iter}")

            return None
        
    genai.configure(api_key=os.environ["API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Write a story about a magic backpack.")
    print(response.text)