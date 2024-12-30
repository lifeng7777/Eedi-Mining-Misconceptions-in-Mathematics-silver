import os

import pandas as pd
import re


df_misconception = pd.read_csv("/data/misconception_mapping.csv")
response_folder = "/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/response/question_1101/"


def extract_data(response):
    respons = re.sub(
        r'(<CorrectAnswer>.*?)(?=(<IncorrectAnswer>|<Thinking>|</ConstructName>|$))',
        lambda m: m.group(1) + '</CorrectAnswer>' if '</CorrectAnswer>' not in m.group(1) else m.group(1),
        response,
        flags=re.DOTALL
    )

    # 同样处理没有闭合标签的 <Incorrect Answer>
    response = re.sub(
        r'(<IncorrectAnswer>.*?)(?=(<Thinking>|</ConstructName>|$))',
        lambda m: m.group(1) + '</IncorrectAnswer>' if '</IncorrectAnswer>' not in m.group(1) else m.group(1),
        respons,
        flags=re.DOTALL
    )
    # 定义正则表达式模式
    # 定义正则表达式模式
    pattern = r'<([^>]+)>(.*?)<\/\1>'

    # 使用正则表达式查找所有匹配的内容
    matches = re.findall(pattern, response, re.DOTALL)

    # 将结果存储到字典中
    data = {match[0]: match[1].replace("$", "").strip() for match in matches}

    return data


df_final = pd.DataFrame()
for i in range(len(df_misconception)):
    misconception = df_misconception.loc[i, 'MisconceptionName']
    misconception_id = df_misconception.loc[i, 'MisconceptionId']
    with open(f"{response_folder}{i}_response.txt", "r") as f:
        response = f.read().strip()
    json_data = extract_data(response)
    if 'tag' in json_data:
        del json_data['tag']
    if len(json_data)!=5:
        os.remove(f"{response_folder}{i}_response.txt")
        continue
    json_data["Misconception"] = misconception
    json_data["MisconceptionId"] = misconception_id

    new_row = pd.DataFrame([json_data])
    df_final = pd.concat([df_final, new_row], ignore_index=True)

df_final.to_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/gpt4o-question_1101.csv", index=False)


