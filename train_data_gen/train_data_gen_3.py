import os.path
import random
from utils.utils import send_message
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager


# 该脚本为每个misconception，采用fewshot的方法，单独生成一个case；统计了misconception出现的次数，出现的次数越少，生成的越多

df_answer_with_misconception = pd.read_csv("/data/train_data_with_misconception.csv")
# df_answer_no_misconception = pd.read_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/train_data_no_misconception_with_thinking.csv")


# 第一步，需要构造一个dict，key是id，value是few-shot的模板
# df_misconception_once = pd.read_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/misconception_once.csv")
# df_misconception_twice = pd.read_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/misconception_twice.csv")
# df_misconception_others = pd.read_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/misconception_others.csv")
# df_misconception_zero = pd.read_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/misconception_zero.csv")
df_misconception = pd.read_csv("/data/misconception_mapping.csv")
# df_misconception = pd.concat([df_misconception_once, df_misconception_twice, df_misconception_others, df_misconception_zero], ignore_index=True)
# 从df_answer_with_misconception里面，先构造一个dict，key是id，value是df_answer_with_misconception的index
id2index = {}
for i in range(len(df_answer_with_misconception)):
    if df_answer_with_misconception.loc[i, 'MisconceptionId'] not in id2index:
        id2index[df_answer_with_misconception.loc[i, 'MisconceptionId']] = [i]
    else:
        id2index[df_answer_with_misconception.loc[i, 'MisconceptionId']].append(i)


# 第三部分，采用few-shot方式进行，对每个Misconception生成3个相关问题
PROMPT_Only_Misconception_few_shot = """
Misconception: {Misconception} 

You are a Mathematics teacher. The above misconception is common errors made by students when solving math problems. 
Your task is to:

1. Speculate on what type of math problems this misconception might occur in, providing:
   - **Construct Name**: The mathematical construct or concept involved
   - **Subject Name**: The subject or sub-field of mathematics
2. Creatively design a possible math problem that could lead to the given misconception.
3. Provide:
   - The correct solution to the problem.
   - The incorrect solution that results from the misunderstanding.

Here are some examples:
{example_prompt}


Your response should radiate creativity and captivate the audience—drawing inspiration from the examples, but steering clear of mirroring them entirely.
Both correct and incorrect answers should be in the format of multiple-choice text. Please ensure that the content you generate aligns with the style and format of a multiple-choice question.
Please provide your answers with the same format as above. Output:
"""




example_prompt = """
Misconception: {Misconception}

<Construct Name>{ConstructName} </Construct Name>
<Subject Name>{SubjectName} </Subject Name>
<Question>{Question} </Question> 
<Correct Answer>{CorrectAnswer} </Correct Answer>
<Incorrect Answer>{IncorrectAnswer} </Incorrect Answer>
"""


PROMPT_Only_Misconception = """

Misconception: {Misconception} 

You are a Mathematics teacher. The above misconception is common errors made by students when solving math problems. 
Your task is to:

1. Speculate on what type of math problems this misconception might occur in, providing:
   - **Construct Name**: The mathematical construct or concept involved
   - **Subject Name**: The subject or sub-field of mathematics
2. Creatively design a possible math problem that could lead to the given misconception.
3. Provide:
   - The correct solution to the problem.
   - The incorrect solution that results from the misunderstanding.
   
Each output field must be between <tag> INSERT TEXT HERE </tag>. Output:"""


def apply_template_fewshot(row):

    prompt = example_prompt.format(
        ConstructName=row["ConstructName"],
        SubjectName=row["SubjectName"],
        Question=row["QuestionText"],
        IncorrectAnswer=row[f"AnswerText"],
        CorrectAnswer=row[f"CorrectAnswerText"],
        Misconception=row["Misconception"])
        # thinking=row['response'])
    return prompt



def process_row(row_index):
    misconception = df_misconception.loc[row_index, 'MisconceptionName']
    id = df_misconception.loc[row_index, 'MisconceptionId']
    if id in id2index:
        return ""
    while id not in id2index:
        id = random.randint(0, len(df_misconception))
    index = id2index[id]
    num_samples = min(5, len(index))
    # 从 index 中随机选择 num_samples 个元素
    random_choices = random.sample(index, num_samples)
    while len(random_choices) < 5:
        random_id = random.randint(0, len(df_answer_with_misconception)-1)
        if random_id not in random_choices:
            random_choices.append(random_id)

    few_shot_template = ""
    for j in random_choices:
        few_shot_template += apply_template_fewshot(df_answer_with_misconception.iloc[j]) + "\n\n"
    final_prompt = PROMPT_Only_Misconception_few_shot.format(Misconception=misconception, example_prompt=few_shot_template)
    save_name = f"/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/response/question_1209/{row_index}_response.txt"
    if os.path.exists(save_name):
        return ""
    else:
        response = send_message(final_prompt)
        df_misconception.loc[i, 'response'] = response
        with open(save_name, "w") as f:
            f.write(response + "\n")
        with open(save_name.replace(".txt", "_prompt.txt"), "w") as f:
            f.write(final_prompt + "\n")

    return response





def parallel_process(df_misconception):
    """
    并行处理 DataFrame，边处理边保存。
    每处理 chunk_size 行数据就保存一次结果到磁盘。
    """
    # 获取需要处理的行数
    num_rows = len(df_misconception)
    print(num_rows)
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


def parallel_preocess_double_check(df_misconception):
    """
    并行处理 DataFrame，边处理边保存。
    每处理 chunk_size 行数据就保存一次结果到磁盘。
    """
    # 获取需要处理的行数
    num_rows = len(df_misconception)
    # 创建一个进程池，大小为 CPU 核心数的一半（或 4，如果核心数少于 8）
    pool_size = min(cpu_count() // 2, 4)

    # 使用 map_async 提交任务
    with Pool(pool_size) as pool:
        # tqdm 用于显示进度条
        # 保存部分批次结果
        for i, result in enumerate(
                tqdm(pool.imap_unordered(double_check, range(num_rows), chunksize=10), total=num_rows)):
            pass

    print("所有数据处理完毕并保存。")


if __name__ == "__main__":
    parallel_process(df_misconception)
    # parallel_preocess_double_check(df_misconception)
    # 遍历读本地文件的response
    for i in range(len(df_misconception)):
        if os.path.exists(f"/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/response/question_1209/{i}_response.txt"):
            with open(f"/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/response/question_1209/{i}_response.txt", 'r') as f:
                response = f.readlines()
            df_misconception.loc[i, 'response'] = "".join(response)
        df_misconception.to_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/train_data_only_misconception_1209.csv")

