import numpy as np
from collections import Counter
import pandas as pd
facebox_csv_root_path = "/kaggle/working/data/casme_2/faceboxcsv"
def remove_outliers(data, m=1.5):
    """
    根据四分位数法则去掉极端值（异常值）
    :param data: 输入列表
    :param m: 异常值的判断因子，默认为1.5倍四分位数间距
    :return: 去除极端值后的列表
    """
    data = np.array(data)
    q1, q3 = np.percentile(data, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - m * iqr
    upper_bound = q3 + m * iqr
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data


def get_most_frequent_average(data):
    """
    统计出现频率最高的数，并计算其平均值
    :param data: 输入列表
    :return: 出现频率最高的数的平均值
    """
    counter = Counter(data)
    # 找到出现频率最高的数
    most_common = counter.most_common()
    highest_frequency = most_common[0][1]
    # 过滤出频率最高的数值
    frequent_numbers = [item for item, freq in most_common if freq == highest_frequency]
    # 计算平均值
    return np.mean(frequent_numbers)

def merged_csv(csv_files_list):
    # 需要合并的 CSV 文件路径列表
    # csv_files = ['file1.csv', 'file2.csv']

    # 读取并合并 CSV 文件
    df_list = [pd.read_csv(file) for file in csv_files_list]
    merged_df = pd.concat(df_list, ignore_index=True)

    # 将合并后的结果写入一个新的 CSV 文件
    merged_df.to_csv('merged_output.csv', index=False)

    print(merged_df)


# def get_sitelist():

# 示例数据
data = [10, 12, 12, 13, 13, 15, 15, 16, 16, 20, 30, 100, 150]

# 去掉极端值
filtered_data = remove_outliers(data)

# 计算剩余数据中频率最高的数的平均值
average_value = get_most_frequent_average(filtered_data)

print("去掉极端值后的数据:", filtered_data)
print("频率最高的数的平均值:", average_value)
