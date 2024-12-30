import pandas as pd
import re
import json
import random

df_prompt = pd.read_parquet("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/submission_question.parquet")
print(len(df_prompt))
print(df_prompt.head())
print(df_prompt.loc[0, 'text'])

# df_label = pd.read_parquet("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/label.parquet")
# print(df_label.head())


df_misconception = pd.read_csv("/data/misconception_mapping.csv")
misconceptions = df_misconception.MisconceptionName.values

def extract_data(text):
    # 正则表达式，用于匹配 system 和 user 的内容
    pattern = r"<\|im_start\|>(system|user)([\s\S]*?)<\|im_end\|>"

    # 提取匹配项
    matches = re.findall(pattern, text)

    # 将提取内容分别保存为 system 和 user 的文本
    result = {"system": [], "user": []}
    for match in matches:
        role, content = match
        result[role].append(content.strip())
    misconception_id = text.split('[')[-1][:-1]

    return result["user"][0], misconception_id.split(',')


llamafactory_save_file = "/data/train_data_with_new_question_ref.json"

llamafactory_res = []
# 对每一条数据，根据misconception_id，找到对应的misconception
cnt = 0
for i in range(len(df_prompt)):
    text = df_prompt.loc[i, 'text']
    prompt, misconception_id = extract_data(text)
    label = str(df_prompt.loc[i, 'QuestionId_Answer']).split('_')[0]
    # 如果label在misconception_id中，则取出对应的misconception的index
    if label in misconception_id:
        index = misconception_id.index(label)
        misconception = df_misconception.loc[int(label), 'MisconceptionName']
        output = f"{index+1}. {misconception}"
    else:
        cnt += 1
        # 随机替换掉一个misconception，先从1～20中随机选一个，然后找到对应的misconception
        random_index = random.randint(1, 20)
        misconception = df_misconception.loc[int(label), 'MisconceptionName']
        misconception_id[random_index-1] = label
        retrival = ""
        for j, ids in enumerate(misconception_id):
            # serial number + misconceptions
            retrival += f'{j + 1}. ' + misconceptions[int(ids)] + '\n'
        prompt = prompt.split("help you make the decision:")[0] + "help you make the decision:\n\n" + retrival
        output = f"{random_index}. {misconception}"

    llamafactory_dict = {
        "instruction": prompt,
        "input": "",
        "system": "",
        "output": output,
        "id": i,
        "misconception_id": label,
        "history": []}
    llamafactory_res.append(llamafactory_dict)

print(f"未召回的数目有:{cnt}")
with open(llamafactory_save_file, 'w') as f:
    f.write(json.dumps(llamafactory_res, indent=2, ensure_ascii=False))