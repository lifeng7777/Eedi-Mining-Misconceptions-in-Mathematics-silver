# 1. 给所有misconception补充thinking（最好的数据）
# 2. 给所有没有misconception的数据, 采用few-shot的方式进行thinking以及misconception的补充（差一点的数据），这一步里面的misconception不在列表里，所以需要一个embedding模型进行对齐
# 3. 通过few-shot的方式, 对每个misconception生成3个相关问题，进行数据集的扩充（差一点的数据）

# 如何训练embedding模型？
# 方式1: 将1中的数据，所有thinking - misconception 作为pair 构建embedding的训练集
# 方式2: 给所有1中的数据，按照2的方式进行推理，然后拿推理结果与真实结果进行对比，构建embedding的训练集（这个应该是在不训练的情况下效果好）
# 得到embedding训练集后，进行训练
# 将训练好的embedding模型，应用到2中的数据上，得到misconception的补充，至此，所有数据都有misconception了

# 训练LLM模型
# 选择32b模型进行lora训练，然后进行awq量化，加载到kaggle上进行推理，推理结果同样使用embedding模型进行对齐，得到最终的结果
import pandas as pd
import json
import re
from tqdm import tqdm




PROMPT_No_Misconception = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.
Before answering the question_1020 think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag.
"""


def apply_template_no_misconception(row):
    prompt = PROMPT_No_Misconception.format(
        ConstructName=row["ConstructName"],
        SubjectName=row["SubjectName"],
        Question=row["QuestionText"],
        IncorrectAnswer=row[f"AnswerText"],
        CorrectAnswer=row[f"CorrectAnswerText"])
    return prompt


def apply_template_only_misconception(row):
    prompt = PROMPT_No_Misconception.format(
        ConstructName=row["ConstructName"],
        SubjectName=row["SubjectName"],
        Question=row["Question"],
        IncorrectAnswer=row[f"IncorrectAnswer"],
        CorrectAnswer=row[f"CorrectAnswer"])
    return prompt


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
    data = {match[0]: match[1].strip() for match in matches}

    return data


llamafactory_res = []
llamafactory_save_file = "/data/mmim_train_data.json"


train_data_file = "/data/train_data_with_misconception_with_thinking.csv"
data = pd.read_csv(train_data_file)
for i in tqdm(range(len(data))):
    prompt = apply_template_no_misconception(data.iloc[i])
    output = data.loc[i, 'response'] + f"\n<response>{data.loc[i, 'Misconception']}</response>"
    llamafactory_dict = {
        "instruction": prompt,
        "input": "",
        "system": "",
        "output": output,
        "history": []}
    llamafactory_res.append(llamafactory_dict)
with open(llamafactory_save_file, 'w') as f:
    f.write(json.dumps(llamafactory_res, indent=2, ensure_ascii=False))


train_data_file = "/data/train_data_no_misconception_with_thinking.csv"
data = pd.read_csv(train_data_file)
for i in tqdm(range(len(data))):
    prompt = apply_template_no_misconception(data.iloc[i])
    output = data.loc[i, 'response']
    llamafactory_dict = {
        "instruction": prompt,
        "input": "",
        "system": "",
        "output": output,
        "history": []}
    llamafactory_res.append(llamafactory_dict)
with open(llamafactory_save_file, 'w') as f:
    f.write(json.dumps(llamafactory_res, indent=2, ensure_ascii=False))



train_data_file = "/data/misconception_mapping.csv"
data = pd.read_csv(train_data_file)
for i in tqdm(range(len(data))):
    with open(f"/response/question_1020/{i}_response_format.txt", 'r') as f:
        response = f.readlines()
    response = "".join(response)
    response = response.replace("Construct Name>", "ConstructName>")
    response = response.replace("Subject Name>", "SubjectName>")
    response = response.replace("Incorrect Answer>", "IncorrectAnswer>")
    response = response.replace("Correct Answer>", "CorrectAnswer>")
    response = response.replace("Question>", "Question>")
    response = response.replace("thinking>", "Thinking>")
    if i==746:
        aaa = 0
    json_data = extract_data(response)
    prompt = apply_template_only_misconception(json_data)
    thinking = json_data["Thinking"]
    output = f"<thinking>{thinking}</thinking>\n" + f"<response>{data.loc[i, 'MisconceptionName']}</response>"
    llamafactory_dict = {
        "instruction": prompt,
        "input": "",
        "system": "",
        "output": output,
        "history": []}
    llamafactory_res.append(llamafactory_dict)
with open(llamafactory_save_file, 'w') as f:
    f.write(json.dumps(llamafactory_res, indent=2, ensure_ascii=False))


