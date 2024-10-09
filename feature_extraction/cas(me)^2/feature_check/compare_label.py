import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from pathlib import Path
import glob
import logging


def load_npz_file(npz_file_path):
    """
    读取npz文件的列
    """
    data = np.load(npz_file_path)
    feature = data['feature']  # 提取特征矩阵
    label = data['label']  # 提取标签矩阵
    video_name = data['video_name']  # 提取视频名称
    return feature, label, video_name


def get_npz_path_list(all_npz_root_path):
    """
    获取所有npz文件
    并将获取的npz文件路径放入一个列表
    """
    all_npz_path_list = []
    for sub_item in Path(all_npz_root_path).iterdir():
        if not sub_item.is_dir():
            continue
        npz_file_path_list = glob.glob(os.path.join(str(sub_item), "*.npz"))
        npz_file_path_list.sort()
        all_npz_path_list.extend(npz_file_path_list)
    return all_npz_path_list


def compare_label_overall(all_npz_root_path1, all_npz_root_path2):
    """
    整体上比较label
    比较video_name label.shape 两个label的内容
    """
    all_npz_path_list1 = get_npz_path_list(all_npz_root_path1)
    all_npz_path_list2 = get_npz_path_list(all_npz_root_path2)
    if len(all_npz_path_list1) != len(all_npz_path_list2):
        print("两个npz文件地址列表长度不同")
        print(f"{all_npz_root_path1}: {len(all_npz_path_list1)}个")
        print(f"{all_npz_root_path2}: {len(all_npz_path_list2)}个")
        return
    for npz_file_path1, npz_file_path2 in zip(all_npz_path_list1, all_npz_path_list2):
        feature, label1, video_name1 = load_npz_file(npz_file_path1)
        npz_file_name1 = os.path.split(npz_file_path1)[-1]
        npz_file_name1 = os.path.splitext(npz_file_name1)[0]
        feature, label2, video_name2 = load_npz_file(npz_file_path1)
        npz_file_name2 = os.path.split(npz_file_path2)[-1]
        npz_file_name2 = os.path.splitext(npz_file_name2)[0]
        if video_name1 != video_name2:
            print("不是同一段视频")
            print(f"npz1: {npz_file_name1}")
            print(f"npz2: {npz_file_name2}")
            continue
        if label1.shape != label2.shape:
            print("两个标签矩阵尺寸不同")
            print(f"npz1: {npz_file_name1} size: {label1.shape}")
            print(f"npz2: {npz_file_name2} size: {label2.shape}")
            continue
        if np.array_equal(label1, label2):
            print("两个标签矩阵内容完全相同")
            print(f"npz1: {npz_file_name1} size:{label1.shape}")
            print(f"npz2: {npz_file_name2} size:{label2.shape}")
        else:
            print("两个标签矩阵内容不完全相同")
            print(f"npz1: {npz_file_name1} size:{label1.shape}")
            print(f"npz2: {npz_file_name2} size:{label2.shape}")


def get_exp_apex_label_tuple(npz_file_name, label):
    # 创建时间轴
    time_axis = np.arange(label.shape[0])

    # 微表情标记
    micro_start_indices = time_axis[label[:, 0] == 1]
    micro_apex_indices = time_axis[label[:, 1] == 1]
    micro_end_indices = time_axis[label[:, 2] == 1]
    micro_action_indices = time_axis[label[:, 3] == 1]

    # 宏表情标记
    macro_start_indices = time_axis[label[:, 4] == 1]
    macro_apex_indices = time_axis[label[:, 5] == 1]
    macro_end_indices = time_axis[label[:, 6] == 1]
    macro_action_indices = time_axis[label[:, 7] == 1]
    micro_exp_apex_tuple = None
    macro_exp_apex_tuple = None
    if len(micro_action_indices) > 0:
        micro_exp_apex_tuple = (
            npz_file_name, micro_start_indices, micro_apex_indices, micro_end_indices, micro_action_indices)
    if len(macro_action_indices) > 0:
        macro_exp_apex_tuple = (
            npz_file_name, macro_start_indices, macro_apex_indices, macro_end_indices, macro_action_indices)

    # 用于核验
    return micro_exp_apex_tuple, macro_exp_apex_tuple


def get_all_label_list(all_npz_root_path):
    # all_npz_root_path train下
    all_micro_exp_apex_list = []
    all_macro_exp_apex_list = []
    for sub_item in Path(all_npz_root_path).iterdir():
        if not sub_item.is_dir():
            continue
        npz_file_path_list = glob.glob(os.path.join(str(sub_item), "*.npz"))
        npz_file_path_list.sort()
        for npz_file_path in npz_file_path_list:
            feature, label, video_name = load_npz_file(npz_file_path)
            npz_file_name = os.path.split(npz_file_path)[-1]
            npz_file_name = os.path.splitext(npz_file_name)[0]
            micro_exp_apex_tuple, macro_exp_apex_tuple = get_exp_apex_label_tuple(npz_file_name, label)
            if micro_exp_apex_tuple:
                all_micro_exp_apex_list.append(micro_exp_apex_tuple)
            if macro_exp_apex_tuple:
                all_macro_exp_apex_list.append(macro_exp_apex_tuple)
    return all_micro_exp_apex_list, all_macro_exp_apex_list


def compare_all_label_list(all_label_list1, all_label_list2):
    # [([])]
    if len(all_label_list1) != len(all_label_list2):
        print("两个标签列表长度不同")
        return False
    all_same_label_list = []
    not_all_same_label_list = []
    # 对每个标签对进行比较
    for subtuple1, subtuple2 in zip(all_label_list1, all_label_list2):
        # 比较分段标签名
        if subtuple1[0] != subtuple2[0] or \
                not np.array_equal(subtuple1[1], subtuple2[1]) or \
                not np.array_equal(subtuple1[2], subtuple2[2]) or \
                not np.array_equal(subtuple1[3], subtuple2[3]) or \
                not np.array_equal(subtuple1[4], subtuple2[4]):
            # 如果标签名或任意帧不相同，加入不完全相同的列表
            not_all_same_label_list.append((subtuple1, subtuple2))
        else:
            # 如果所有字段相同，加入完全相同的列表
            all_same_label_list.append(subtuple1)
    return not_all_same_label_list, all_same_label_list


def output_all_label(not_all_same_micro_label_list, all_same_micro_label_list, not_all_same_macro_label_list,
                     all_same_macro_label_list):

    print(f"微表情分段标签 不完全相同的 总共{len(not_all_same_micro_label_list)}对")
    if len(not_all_same_micro_label_list):
        for item in not_all_same_micro_label_list:
            npz_file_name1, micro_start_indices1, micro_apex_indices1, micro_end_indices1, micro_action_indices1 = item[
                0]
            npz_file_name2, micro_start_indices2, micro_apex_indices2, micro_end_indices2, micro_action_indices2 = item[
                1]
            print("*******************************************************")
            print(f"分段标签 {npz_file_name1}")
            print(f"起始帧: {micro_start_indices1}    顶点帧: {micro_apex_indices1}    偏移帧: {micro_end_indices1}")
            print(f"微表情区间: {micro_action_indices1}")
            print(f"分段标签 {npz_file_name2}")
            print(f"起始帧: {micro_start_indices2}    顶点帧: {micro_apex_indices2}    偏移帧: {micro_end_indices2}")
            print(f"微表情区间: {micro_action_indices2}")
            print()  # 输出一个换行
    print("-----------------------------------------------------------------------------------------------------------")

    print(f"微表情分段标签 完全相同的 总共{len(all_same_micro_label_list)}对")
    if len(all_same_micro_label_list):
        for item in all_same_micro_label_list:
            npz_file_name, micro_start_indices, micro_apex_indices, micro_end_indices, micro_action_indices = item
            print(f"分段标签 {npz_file_name}")
            print(f"起始帧: {micro_start_indices}    顶点帧: {micro_apex_indices}    偏移帧: {micro_end_indices}")
            print(f"微表情区间: {micro_action_indices}")
            print("")  # 输出一个换行
    print("-----------------------------------------------------------------------------------------------------------")

    print(f"宏表情分段标签 不完全相同的 总共{len(not_all_same_macro_label_list)}对")
    if len(not_all_same_macro_label_list):
        for item in not_all_same_macro_label_list:
            npz_file_name1, macro_start_indices1, macro_apex_indices1, macro_end_indices1, macro_action_indices1 = item[
                0]
            npz_file_name2, macro_start_indices2, macro_apex_indices2, macro_end_indices2, macro_action_indices2 = item[
                1]
            print("*******************************************************")
            print(f"分段标签 {npz_file_name1}")
            print(f"起始帧: {macro_start_indices1}    顶点帧: {macro_apex_indices1}    偏移帧: {macro_end_indices1}")
            print(f"宏表情区间: {macro_action_indices1}")
            print(f"分段标签 {npz_file_name2}")
            print(f"起始帧: {macro_start_indices2}    顶点帧: {macro_apex_indices2}    偏移帧: {macro_end_indices2}")
            print(f"宏表情区间: {macro_action_indices2}")
            print()  # 输出一个换行
    print("-----------------------------------------------------------------------------------------------------------")

    print(f"宏表情分段标签 完全相同的 总共{len(all_same_macro_label_list)}对")
    if len(all_same_macro_label_list):

        for item in all_same_macro_label_list:
            npz_file_name, macro_start_indices, macro_apex_indices, macro_end_indices, macro_action_indices = item
            print(f"分段标签 {npz_file_name}")
            print(f"起始帧: {macro_start_indices}    顶点帧: {macro_apex_indices}    偏移帧: {macro_end_indices}")
            print(f"宏表情区间: {macro_action_indices}")
            print("")  # 输出一个换行
    print("-----------------------------------------------------------------------------------------------------------")


def check_all_label(all_npz_root_path1, all_npz_root_path2):
    # all_npz_root_path1 = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/"
    all_micro_exp_apex_list1, all_macro_exp_apex_list1 = get_all_label_list(all_npz_root_path1)
    # print("-----------------------------------------------------------------------------------------------------------")
    # all_npz_root_path2 = "D:/PycharmProjects/ME-GCN-Project/feature_segment_apex/train"
    all_micro_exp_apex_list2, all_macro_exp_apex_list2 = get_all_label_list(all_npz_root_path2)

    not_all_same_micro_label_list, all_same_micro_label_list = compare_all_label_list(all_micro_exp_apex_list1,
                                                                                      all_micro_exp_apex_list2)
    not_all_same_macro_label_list, all_same_macro_label_list = compare_all_label_list(all_macro_exp_apex_list1,
                                                                                      all_macro_exp_apex_list2)
    output_all_label(not_all_same_micro_label_list, all_same_micro_label_list, not_all_same_macro_label_list,
                     all_same_macro_label_list)


if __name__ == "__main__":
    # # 设置日志文件路径
    # log_file_path = "./logfile.log"
    # # 初始化日志记录
    # setup_logging(log_file_path)
    all_npz_root_path1 = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/test"
    all_npz_root_path2 = "D:/PycharmProjects/ME-GCN-Project/feature_segment_apex_1/test/"

    print("总体上的比较")
    compare_label_overall(all_npz_root_path1, all_npz_root_path2)

    print("具体 微表情 宏表情 比较")
    check_all_label(all_npz_root_path1, all_npz_root_path2)

