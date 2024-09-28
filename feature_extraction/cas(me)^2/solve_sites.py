import numpy as np
from collections import Counter
import pandas as pd
from pathlib import Path
import os
import glob
import csv

facebox_csv_root_path = "/kaggle/working/data/casme_2/faceboxcsv"


def remove_outliers(data, m=1.5):
    """
    根据四分位数法则去掉极端值（异常值）
    :param data: 输入列表
    :param m: 异常值的判断因子，默认为1.5倍四分位数间距
    :return: 去除极端值后的列表
    """
    # data = np.array(data)
    # q1, q3 = np.percentile(data, [25, 75])
    # iqr = q3 - q1
    # lower_bound = q1 - m * iqr
    # upper_bound = q3 + m * iqr
    # filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    # return filtered_data
    # 假设简单去除离群点的方法：根据四分位距过滤
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - m * iqr
    upper_bound = q3 + m * iqr
    return [x for x in data if lower_bound <= x <= upper_bound]


def get_most_frequent_average(data):
    """
    统计出现频率最高的数，并计算其平均值
    :param data: 输入列表
    :return: 出现频率最高的数的平均值
    """
    # counter = Counter(data)
    # # 找到出现频率最高的数
    # most_common = counter.most_common()
    # highest_frequency = most_common[0][1]
    # # 过滤出频率最高的数值
    # frequent_numbers = [item for item, freq in most_common if freq == highest_frequency]
    # # 计算平均值
    # return np.mean(frequent_numbers)
    # 假设获取出现频率最高的数的平均值
    from collections import Counter
    counter = Counter(data)
    most_common_values = [k for k, v in counter.items() if v == counter.most_common(1)[0][1]]
    return sum(most_common_values) / len(most_common_values)


def example():
    """
    用于测试
    """
    # 示例数据
    data = [10, 12, 12, 13, 13, 15, 15, 16, 16, 20, 30, 100, 150]

    # 去掉极端值
    filtered_data = remove_outliers(data)

    # 计算剩余数据中频率最高的数的平均值
    average_value = get_most_frequent_average(filtered_data)

    print("去掉极端值后的数据:", filtered_data)
    print("频率最高的数的平均值:", average_value)


def merged_csv(csv_files_list, output_root_path):
    """
        合并多个 CSV 文件，跳过空文件。如果列表中只有一个文件，且非空，直接处理该文件。

        参数:
            csv_files_list (list): 需要合并的 CSV 文件路径列表。
            output_root_path (str): 合并后的 CSV 文件保存路径。

        返回:
            str: 合并后的 CSV 文件路径。
        """
    df_list = []

    # 遍历 CSV 文件列表，读取非空文件
    for file in csv_files_list:
        try:
            df = pd.read_csv(file, header=None)
            # 仅在文件非空时添加到列表中
            if not df.empty:
                df_list.append(df)
        except pd.errors.EmptyDataError:
            print(f"文件 {file} 为空，已跳过。")

    # 如果没有可合并的文件，返回 None
    if not df_list:
        print("所有文件都为空或无可用数据，无法进行合并。")
        return None
    # 如果只有一个非空文件，直接保存它
    if len(df_list) == 1:
        output_csv_path = os.path.join(output_root_path, 'merged_output.csv')
        df_list[0].to_csv(output_csv_path, index=False)
        return output_csv_path

    print(output_root_path)

    # 合并多个文件
    # 按行合并
    merged_df = pd.concat(df_list, ignore_index=True)
    # 将合并后的结果写入一个新的 CSV 文件
    output_csv_path = os.path.join(output_root_path, 'merged_output.csv')
    merged_df.to_csv(output_csv_path, index=False)
    return output_csv_path


def record_csv(csv_path, rows):
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w') as f:
        csv_w = csv.writer(f)
        csv_w.writerows(rows)


def get_facebox_average():
    for sub_item in Path(facebox_csv_root_path).iterdir():
        if not sub_item.is_dir():
            continue

        for type_item in sub_item.iterdir():
            if not type_item.is_dir():
                continue
            csv_path_list = glob.glob(os.path.join(str(type_item), "*.csv"))
            output_root_path = os.path.join(facebox_csv_root_path, sub_item.name, type_item.name)
            output_csv_path = None
            if csv_path_list:
                output_csv_path = merged_csv(csv_path_list, output_root_path)
            if output_csv_path and os.path.exists(output_csv_path):
                df = pd.read_csv(output_csv_path, header=None)
                # 去掉极端值
                filtered_data_left = remove_outliers(df[0].tolist())
                # 计算剩余数据中频率最高的数的平均值
                average_value_left = get_most_frequent_average(filtered_data_left)
                # 去掉极端值
                filtered_data_top = remove_outliers(df[1].tolist())
                # 计算剩余数据中频率最高的数的平均值
                average_value_top = get_most_frequent_average(filtered_data_top)
                # 去掉极端值
                filtered_data_right = remove_outliers(df[2].tolist())
                # 计算剩余数据中频率最高的数的平均值
                average_value_right = get_most_frequent_average(filtered_data_right)
                # 去掉极端值
                filtered_data_bottom = remove_outliers(df[3].tolist())
                # 计算剩余数据中频率最高的数的平均值
                average_value_bottom = get_most_frequent_average(filtered_data_bottom)

                facebox_average = [(average_value_left, average_value_top, average_value_right, average_value_bottom)]

                print(facebox_average)
                facebox_average_path = os.path.join(output_root_path, "facebox_average.csv")
                record_csv(facebox_average_path, facebox_average)


if __name__ == "__main__":
    get_facebox_average()
