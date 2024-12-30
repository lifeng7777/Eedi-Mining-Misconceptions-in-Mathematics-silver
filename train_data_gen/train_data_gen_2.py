from utils.utils import send_message
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import os

# 该脚本为没有misconception的数据添加thinking和misconception

df_answer_with_misconception = pd.read_csv("../data/train_data_with_misconception_with_thinking.csv")
df_answer_no_misconception = pd.read_csv("../data/train_data_no_misconception.csv")


# 第一部分，有misconception的部分数据
if 'response' not in df_answer_no_misconception.columns:
    df_answer_no_misconception['response'] = None
# 第二部分，没有misconception的部分数据，采用few-shot方式进行
PROMPT_No_Misconception = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.
Before answering the question_1020 think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag.
{thinking}
<response>{Misconception}</response>
"""


def apply_template_no_misconception(row):
    prompt = PROMPT_No_Misconception.format(
        ConstructName=row["ConstructName"],
        SubjectName=row["SubjectName"],
        Question=row["QuestionText"],
        IncorrectAnswer=row[f"AnswerText"],
        CorrectAnswer=row[f"CorrectAnswerText"],
        Misconception=row["Misconception"],
        thinking=row['response'])
    return prompt


def create_few_shot_template():
    few_shot_template = ""
    for i in range(3):
        few_shot_template += apply_template_no_misconception(df_answer_with_misconception.iloc[i]) + "\n\n"
    return few_shot_template


few_shot_template = create_few_shot_template()


def process_row(row_index,):
    row = df_answer_no_misconception.iloc[row_index]
    prompt = few_shot_template + row['Prompt']
    if pd.isna(row['response']):
        response = send_message(prompt)
        df_answer_no_misconception.loc[row_index, 'response'] = response
        # 保存text到文件
        with open(f"/response/thinking_and_misconception/{row_index}_response.txt", "w") as f:
            f.write(response + "\n")
    return response


def parallel_process(df_answer_no_misconception):
    """
    并行处理 DataFrame，边处理边保存。
    每处理 chunk_size 行数据就保存一次结果到磁盘。
    """
    # 获取需要处理的行数
    num_rows = len(df_answer_no_misconception)
    # 创建一个进程池，大小为 CPU 核心数的一半（或 4，如果核心数少于 8）
    pool_size = min(cpu_count() // 2, 4)

    # 使用 map_async 提交任务
    with Pool(pool_size) as pool:
        # tqdm 用于显示进度条
        # 保存部分批次结果
        for i, result in enumerate(
                tqdm(pool.imap_unordered(process_row, range(num_rows), chunksize=10), total=num_rows)):
            pass

    print("所有数据处理完毕并保存。")

if __name__ == "__main__":
    parallel_process(df_answer_no_misconception)
    # 遍历读本地文件的response
    for i in range(len(df_answer_no_misconception)):
        with open(f"./response/thinking_and_misconception/{i}_response.txt", 'r') as f:
            response = f.readlines()
        df_answer_no_misconception.loc[i, 'response'] = "".join(response)
    df_answer_no_misconception.to_csv("./data/train_data_no_misconception_with_thinking.csv")
