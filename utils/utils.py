import os
from openai import OpenAI

# openai.api_key = "sk-xxxx"  # 请替换为你的实际 API 密钥
client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-xxxx"
)

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key="sk-xxxx",
#     base_url="https://api.chatanywhere.tech/v1"
# )


def send_message(content):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content,
            }
        ],
        model="o1-mini"
    )
    return chat_completion.choices[0].message.content

