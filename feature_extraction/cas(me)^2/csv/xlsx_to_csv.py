import os
import glob
import shutil
from pathlib import Path

import yaml
import numpy as np
import pandas as pd


def get_subject_dict(anno_file):
    """
    输出：
    生成的字典：
    {'1': '15', '2': '16', '3': '19', '4': '20', '5': '21',
        '6': '22', '7': '23', '8': '24', '9': '25', '10': '26',
        '11': '27', '12': '29', '13': '30', '14': '31', '15': '32',
        '16': '33', '17': '34', '18': '35', '19': '36', '20': '37',
        '21': '38', '22': '40'}

    """
    # 读取Excel文件
    xl = pd.ExcelFile(anno_file)
    # 获取第二个工作表
    nameing_rule_1 = xl.parse(xl.sheet_names[1], header=None, dtype=str)
    # 获取第一列的数据并转换为列表
    first_column_list = nameing_rule_1.iloc[:, 0].tolist()  # 获取第一列并转换为列表
    # 获取第三列的数据并转换为列表
    third_column_list = nameing_rule_1.iloc[:, 2].tolist()  # 获取第三列并转换为列表
    subject_dict = {}
    # 检查长度是否一致
    if len(first_column_list) == len(third_column_list):
        # 创建字典，第三列作为键，第一列作为值
        subject_dict = {third_column_list[i]: first_column_list[i] for i in range(len(third_column_list))}
        # 打印字典
        # print("生成的字典：")
        # print(subject_dict)
    else:
        print("第一列和第三列的长度不一致！")

    return subject_dict


def get_type_dict(anno_file):
    """
    输出
    生成的字典：
    {'disgust1': '0101', 'disgust2': '0102', 'anger1': '0401',
        'anger2': '0402', 'happy1': '0502', 'happy2': '0503',
        'happy3': '0505', 'happy4': '0507', 'happy5': '0508'}


    """
    # 读取Excel文件
    xl = pd.ExcelFile(anno_file)
    # 获取第三个工作表
    nameing_rule_2 = xl.parse(xl.sheet_names[2], header=None, dtype=str)
    # 获取第一列的数据并转换为列表
    first_column_list = nameing_rule_2.iloc[:, 0].tolist()  # 获取第一列并转换为列表
    # 获取第二列的数据并转换为列表
    second_column_list = nameing_rule_2.iloc[:, 1].tolist()  # 获取第三列并转换为列表
    type_dict = {}
    # 检查长度是否一致
    if len(first_column_list) == len(second_column_list):
        # 创建字典，第二列作为键，第一列作为值
        type_dict = {second_column_list[i]: first_column_list[i] for i in range(len(second_column_list))}
        # 打印字典
        # print("生成的字典：")
        # print(type_dict)
    else:
        print("第一列和第二列的长度不一致！")

    return type_dict


def parse_code_final(anno_file):
    """
    首先需要处理
    规范csv文件的格式
    pip install openpyxl
    """
    # 读取subject字典
    subject_dict = get_subject_dict(anno_file)
    # 读取type字典
    type_dict = get_type_dict(anno_file)
    # 读取Excel文件
    xl = pd.ExcelFile(anno_file)
    # 获取第一个工作表
    # 第一行为数据 不是列名
    CASFEcode_final = xl.parse(xl.sheet_names[0], header=None, dtype=str)
    # 获取第一列的数据
    first_column_list = CASFEcode_final.iloc[:, 0].tolist()
    # 遍历第一列，并根据 subject_dict 进行修改
    for idx, item in enumerate(first_column_list):
        # 判断 item 是否在 subject_dict 中
        if item in subject_dict.keys():
            # 替换当前项为格式化的字符串
            first_column_list[idx] = f"casme_0{subject_dict[item]}"
        else:
            print(item)
    # 将修改后的第一列写回到 DataFrame 中
    CASFEcode_final.iloc[:, 0] = first_column_list

    # 获取第二列的数据
    second_column_list = CASFEcode_final.iloc[:, 1].tolist()
    # 遍历第二列，并根据 type_dict 进行修改
    for idx, item in enumerate(second_column_list):
        # 判断 item.split('_')[0] 是否在 type_dict 中
        item = item.split('_')[0]
        if item in type_dict.keys():
            # 替换当前项为格式化的字符串
            second_column_list[idx] = f"{first_column_list[idx]}_{type_dict[item]}"
        else:
            print(item)
    # 将修改后的第二列写回到 DataFrame 中
    CASFEcode_final.iloc[:, 1] = second_column_list

    # 获取第六列的数据
    sixth_column_list = CASFEcode_final.iloc[:, 5].tolist()
    # 将第六列写到第七列 DataFrame 中
    CASFEcode_final.iloc[:, 6] = sixth_column_list

    # 获取第八列的数据
    eighth_column_list = CASFEcode_final.iloc[:, 7].tolist()
    # 遍历第八列，进行替换修改
    for idx, item in enumerate(eighth_column_list):
        if item == "macro-expression":
            eighth_column_list[idx] = "1"
        elif item == "micro-expression":
            eighth_column_list[idx] = "2"
        else:
            print(item)
    # 将修改后的第八列写回到第六列 DataFrame 中
    CASFEcode_final.iloc[:, 5] = eighth_column_list
    # 删除第八列（第7索引）和第九列（第8索引）
    CASFEcode_final.drop(CASFEcode_final.columns[[7, 8]], axis=1, inplace=True)
    # 将列名写入 csv
    col_name_list = ["subject", "video_name", "start_frame", "apex_frame", "end_frame", "type_idx", "au"]
    CASFEcode_final.columns = col_name_list

    if os.path.exists("./cas(me)^2_original.csv"):
        os.remove("./cas(me)^2_original.csv")
    CASFEcode_final.to_csv('./cas(me)^2_original.csv', index=False)


if __name__ == "__main__":
    anno_file = "D:/PycharmProjects/ME-GCN-Project/feature_extraction/cas(me)^2/csv/CAS(ME)^2code_final(Updated).xlsx"
    parse_code_final(anno_file)

