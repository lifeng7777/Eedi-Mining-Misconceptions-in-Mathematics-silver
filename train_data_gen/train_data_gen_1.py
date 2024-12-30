from utils.utils import send_message
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, Manager
import os


# 该脚本为有misconception的case生产thinking过程

output_file_path = "../data/train_data_with_misconception_with_thinking.csv"
df_answer_with_misconception = pd.read_csv("../data/train_data_with_misconception.csv")

# 第一部分，有misconception的部分数据
if 'response' not in df_answer_with_misconception.columns:
    df_answer_with_misconception['response'] = None

def process_row(row_index):
    row = df_answer_with_misconception.iloc[row_index]
    prompt = row['Prompt']
    if pd.isna(row['response']):
        response = send_message(prompt)
        df_answer_with_misconception.loc[row_index, 'response'] = response
        # 保存text到文件
        with open(f"./response/thinking/{row_index}_response.txt", "w") as f:
            f.write(response + "\n")


def save_dataframe_partial(output_file, batch_data):
    """
    保存部分批量处理完成的数据到 CSV。
    如果文件已存在，使用 append 模式。
    """
    # 将处理批的数据转为 DataFrame
    df_partial = pd.DataFrame(batch_data)

    # 如果文件不存在，写入 header 作为新文件；否则追加数据并不写入 header
    if not os.path.isfile(output_file):
        df_partial.to_csv(output_file, index=False, mode='w', header=True)
    else:
        df_partial.to_csv(output_file, index=False, mode='a', header=False)
    print(f"已保存 {len(batch_data)} 条数据到 {output_file}")


def parallel_process(df_answer_with_misconception):
    """
    并行处理 DataFrame，边处理边保存。
    每处理 chunk_size 行数据就保存一次结果到磁盘。
    """
    output_file = output_file_path
    # 获取需要处理的行数
    num_rows = len(df_answer_with_misconception)

    # 多进程安全地存储结果
    manager = Manager()
    processed_rows = manager.list()  # 使用 Manager 创建一个共享列表

    # 创建一个进程池，大小为 CPU 核心数的一半（或 4，如果核心数少于 8）
    pool_size = min(cpu_count() // 2, 4)

    # 使用 map_async 提交任务
    with Pool(pool_size) as pool:
        # tqdm 用于显示进度条
        # 保存部分批次结果
        for i, result in enumerate(
                tqdm(pool.imap_unordered(process_row, range(num_rows), chunksize=10), total=num_rows)):
            pass

    # 确保处理完所有数据后，保存最后剩余的数据
    if len(processed_rows) > 0:
        save_dataframe_partial(output_file, list(processed_rows))

    print("所有数据处理完毕并保存。")


if __name__ == "__main__":
    parallel_process(df_answer_with_misconception)
    # 遍历读本地文件的response
    for i in range(len(df_answer_with_misconception)):
        with open(f"./response/thinking/{i}_response.txt", 'r') as f:
            response = f.readlines()
        df_answer_with_misconception.loc[i, 'response'] = "".join(response)
    df_answer_with_misconception.to_csv("../data/train_data_with_misconception_with_thinking.csv")


