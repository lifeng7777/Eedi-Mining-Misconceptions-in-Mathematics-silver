import pandas as pd
import re


# 第一类prompt数据，根据官方给的训练数据进行；有标签的，补充思维链，没有标签的，补充思维链+标签
# 补充后的标签，使用相似度进行embedding模型训练，得到真正的标签

# 第二类prompt数据，根据官方给的误解，让大模型给出一个问题的例子，同时给出思维链。

# 获取所有训练数据后，sft一个大模型，然后通过大模型给出思维链，然后再给出标签，然后使用embedding模型关联到真正的标签


df_train = pd.read_csv("/data/train.csv")
df_mapping = pd.read_csv("/data/misconception_mapping.csv")
misconception_dict = dict(zip(df_mapping.iloc[:, 0], df_mapping.iloc[:, 1]))

PROMPT = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Misconception: {Misconception}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Identify the misconception in the respondent's reasoning that led to the incorrect answer. Use chain-of-thought reasoning to break down their thought process step by step. Ensure that the reasoning is concise, remaining within 100 words, and all details are contained within a single <thinking> tag."""


PROMPT_No_Misconception = """Question: {Question}
Incorrect Answer: {IncorrectAnswer}
Correct Answer: {CorrectAnswer}
Construct Name: {ConstructName}
Subject Name: {SubjectName}

Your task: Identify the misconception behind Incorrect Answer. Answer concisely and generically inside <response>$$INSERT TEXT HERE$$</response>.
Before answering the question_1020 think step by step concisely in 1-2 sentence inside <thinking>$$INSERT TEXT HERE$$</thinking> tag and respond your final misconception inside <response>$$INSERT TEXT HERE$$</response> tag."""


def apply_template(row):
    prompt = PROMPT.format(
                 ConstructName=row["ConstructName"],
                 SubjectName=row["SubjectName"],
                 Question=row["QuestionText"],
                 IncorrectAnswer=row[f"AnswerText"],
                 CorrectAnswer=row[f"CorrectAnswerText"],
                 Misconception=row["Misconception"])
    return prompt


def apply_template_no_misconception(row):
    prompt = PROMPT_No_Misconception.format(
                 ConstructName=row["ConstructName"],
                 SubjectName=row["SubjectName"],
                 Question=row["QuestionText"],
                 IncorrectAnswer=row[f"AnswerText"],
                 CorrectAnswer=row[f"CorrectAnswerText"])
    return prompt


def get_correct_answer(row):
    if row['CorrectAnswer'] == 'A':
        return row['AnswerAText']
    elif row['CorrectAnswer'] == 'B':
        return row['AnswerBText']
    elif row['CorrectAnswer'] == 'C':
        return row['AnswerCText']
    elif row['CorrectAnswer'] == 'D':
        return row['AnswerDText']
    else:
        return None


# Apply the function to create the CorrectAnswer column
df_train['CorrectAnswerText'] = df_train.apply(get_correct_answer, axis=1)
select_column = ["QuestionId", "ConstructName", "SubjectName", "CorrectAnswer", "QuestionText", "CorrectAnswerText"]
# 选择 Answer 和 Misconception 列的映射
answer_columns = [f"Answer{ans}Text" for ans in ["A", "B", "C", "D"]]
misconception_columns = [f"Misconception{ans}Id" for ans in ["A", "B", "C", "D"]]

# 对 AnswerText 和 MisconceptionId 同时执行 melt 操作
df_answer = pd.melt(
    df_train,
    id_vars=select_column,
    value_vars=answer_columns,
    var_name="Option",
    value_name="AnswerText"
)

# 对 MisconceptionId 同步进行 melt 操作
df_misconception = pd.melt(
    df_train,
    id_vars=select_column,
    value_vars=misconception_columns,
    var_name="Option",
    value_name="MisconceptionId"
)

# 合并 AnswerText 和 MisconceptionId 的结果
df_answer['MisconceptionId'] = df_misconception['MisconceptionId']
df_answer['Option'] = df_answer['Option'].apply(lambda x: re.search(r'Answer([A-D])', x).group(1) if re.search(r'Answer([A-D])', x) else None)

# 分为两部分，一部分MisconceptionId为nan，另一部分不为nan

df_answer_with_misconception = df_answer[~df_answer['MisconceptionId'].isna()]
df_answer_with_misconception["Misconception"] = df_answer_with_misconception['MisconceptionId'].apply(lambda x: misconception_dict[x])
df_answer_with_misconception["Prompt"] = df_answer_with_misconception.apply(lambda row: apply_template(row), axis=1)
df_answer_with_misconception = df_answer_with_misconception[df_answer_with_misconception['CorrectAnswer'] != df_answer_with_misconception['Option']]
df_answer_with_misconception = df_answer_with_misconception.reset_index(drop=True)
df_answer_with_misconception.to_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/train_data_with_misconception.csv", index=False)

df_answer_no_misconception = df_answer[df_answer['MisconceptionId'].isna()]
# df_answer_no_misconception["Misconception"] = df_answer_with_misconception['MisconceptionId'].apply(lambda x: misconception_dict[x])
df_answer_no_misconception["Prompt"] = df_answer_no_misconception.apply(lambda row: apply_template_no_misconception(row), axis=1)
df_answer_no_misconception = df_answer_no_misconception[df_answer_no_misconception['CorrectAnswer'] != df_answer_no_misconception['Option']]
df_answer_no_misconception = df_answer_no_misconception.reset_index(drop=True)
df_answer_no_misconception.to_csv("/Users/lifeng/PycharmProjects/MiningMisconceptionsInMathematics/data/train_data_no_misconception.csv", index=False)



print(len(df_answer), len(df_answer_with_misconception), len(df_answer_no_misconception))


# 第一步，将所有思维链补全（弄完拿几个prompt，当作fewshot给df_answer_no_misconception参考）



