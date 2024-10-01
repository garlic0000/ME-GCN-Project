import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from pathlib import Path
import glob


def load_npz_file(npz_file_path):
    data = np.load(npz_file_path)
    feature = data['feature']  # 提取特征矩阵
    label = data['label']  # 提取标签矩阵
    video_name = data['video_name']  # 提取视频名称
    return feature, label, video_name


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
        # print(f"分段特征 {npz_file_name}")
        # print("微表情")
        # print(f"起始帧: {micro_start_indices}    顶点帧: {micro_apex_indices}    偏移帧: {micro_end_indices}")
        # print(f"微表情区间: {micro_action_indices}")
        # print("\n")
    if len(macro_action_indices) > 0:
        macro_exp_apex_tuple = (
            npz_file_name, macro_start_indices, macro_apex_indices, macro_end_indices, macro_action_indices)
        # print(f"分段特征 {npz_file_name}")
        # print("宏表情")
        # print(f"起始帧: {macro_start_indices}    顶点帧: {macro_apex_indices}    偏移帧: {macro_end_indices}")
        # print(f"宏表情区间: {macro_action_indices}")
        # print("\n")

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
            # print("Video Name:", video_name)
            # print("Feature shape:", feature.shape)
            # print("Label shape:", label.shape)
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
    for subtuple1, subtuple2 in zip(all_label_list1, all_label_list2):
        # 分段标签名 字符串
        if subtuple1[0] != subtuple2[0]:
            not_all_same_label_list.append((subtuple1, subtuple2))
            continue
        # 起始帧 np数组
        if subtuple1[1].all() != subtuple2[1].all():
            not_all_same_label_list.append((subtuple1, subtuple2))
            continue
        # 顶点帧
        if subtuple1[2].all() != subtuple2[2].all():
            not_all_same_label_list.append((subtuple1, subtuple2))
            continue
        # 结束帧
        if subtuple1[3].all() != subtuple2[3].all():
            not_all_same_label_list.append((subtuple1, subtuple2))
            continue
        # 动作区间
        if subtuple1[4].all() != subtuple2[4].all():
            not_all_same_label_list.append((subtuple1, subtuple2))
            continue
        all_same_label_list.append(subtuple1)
    return not_all_same_label_list, all_same_label_list


def print_all_label(not_all_same_micro_label_list, all_same_micro_label_list, not_all_same_macro_label_list,
                    all_same_macro_label_list):
    if len(not_all_same_micro_label_list):
        print(f"微表情分段标签 不完全相同 总共{len(not_all_same_micro_label_list)}对")
        for item in not_all_same_micro_label_list:
            npz_file_name1, micro_start_indices1, micro_apex_indices1, micro_end_indices1, micro_action_indices1 = item[
                0]
            npz_file_name2, micro_start_indices2, micro_apex_indices2, micro_end_indices2, micro_action_indices2 = item[
                1]
            print("*******************************************************")
            print(f"分段特征 {npz_file_name1}")
            print(f"起始帧: {micro_start_indices1}    顶点帧: {micro_apex_indices1}    偏移帧: {micro_end_indices1}")
            print(f"微表情区间: {micro_action_indices1}")
            print(f"分段特征 {npz_file_name2}")
            print(f"起始帧: {micro_start_indices2}    顶点帧: {micro_apex_indices2}    偏移帧: {micro_end_indices2}")
            print(f"微表情区间: {micro_action_indices2}")
            print()  # 输出一个换行
    print("-----------------------------------------------------------------------------------------------------------")

    if len(all_same_micro_label_list):
        print(f"微表情分段标签 完全相同 总共{len(all_same_micro_label_list)}个")
        for item in all_same_micro_label_list:
            npz_file_name, micro_start_indices, micro_apex_indices, micro_end_indices, micro_action_indices = item
            print(f"分段特征 {npz_file_name}")
            print(f"起始帧: {micro_start_indices}    顶点帧: {micro_apex_indices}    偏移帧: {micro_end_indices}")
            print(f"微表情区间: {micro_action_indices}")
            print()  # 输出一个换行
    print("-----------------------------------------------------------------------------------------------------------")

    if len(not_all_same_macro_label_list):
        print(f"宏表情分段标签 不完全相同 总共{len(not_all_same_macro_label_list)}对")
        for item in not_all_same_macro_label_list:
            npz_file_name1, macro_start_indices1, macro_apex_indices1, macro_end_indices1, macro_action_indices1 = item[
                0]
            npz_file_name2, macro_start_indices2, macro_apex_indices2, macro_end_indices2, macro_action_indices2 = item[
                1]
            print("*******************************************************")
            print(f"分段特征 {npz_file_name1}")
            print(f"起始帧: {macro_start_indices1}    顶点帧: {macro_apex_indices1}    偏移帧: {macro_end_indices1}")
            print(f"宏表情区间: {macro_action_indices1}")
            print(f"分段特征 {npz_file_name2}")
            print(f"起始帧: {macro_start_indices2}    顶点帧: {macro_apex_indices2}    偏移帧: {macro_end_indices2}")
            print(f"宏表情区间: {macro_action_indices2}")
            print()  # 输出一个换行
    print("-----------------------------------------------------------------------------------------------------------")

    if len(all_same_macro_label_list):
        print(f"宏表情分段标签 完全相同 总共{len(all_same_macro_label_list)}个")
        for item in all_same_macro_label_list:
            npz_file_name, macro_start_indices, macro_apex_indices, macro_end_indices, macro_action_indices = item
            print(f"分段特征 {npz_file_name}")
            print(f"起始帧: {macro_start_indices}    顶点帧: {macro_apex_indices}    偏移帧: {macro_end_indices}")
            print(f"宏表情区间: {macro_action_indices}")
            print()  # 输出一个换行
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
    print_all_label(not_all_same_micro_label_list, all_same_micro_label_list, not_all_same_macro_label_list,
                    all_same_macro_label_list)


def get_all_npz_path_list(all_npz_root_path):
    all_npz_root_path_list = []
    for sub_item in Path(all_npz_root_path).iterdir():
        if not sub_item.is_dir():
            continue
        npz_file_path_list = glob.glob(os.path.join(str(sub_item), "*.npz"))
        npz_file_path_list.sort()
        all_npz_root_path_list.extend(npz_file_path_list)
    all_npz_root_path_list.sort()
    return all_npz_root_path_list


def compare_feature_numpy(npz_file_path1, npz_file_path2):
    feature1, label1, video_name1 = load_npz_file(npz_file_path1)
    feature2, label2, video_name2 = load_npz_file(npz_file_path2)
    npz_file_name1 = os.path.split(npz_file_path1)[-1]
    npz_file_name2 = os.path.split(npz_file_path2)[-1]
    not_same_npz_file_name_list = []
    same_split_feature_name_list = []
    not_same_split_feature_name_list = []
    if npz_file_name1 != npz_file_name2:
        # print(f"不是同一分段特征")
        not_same_npz_file_name_list.append((npz_file_name1, npz_file_name2))
    npz_file_name = os.path.splitext(npz_file_name1)[0]
    if feature1.all() == feature2.all():
        # print(f"{npz_file_name} 分段特征相同")
        same_split_feature_name_list.append(npz_file_name)
    else:
        # print(f"{npz_file_name} 分段特征不同")
        not_same_split_feature_name_list.append(npz_file_name)

    return not_same_npz_file_name_list, same_split_feature_name_list, not_same_split_feature_name_list


def compare_all_feature_numpy(all_npz_path_list1, all_npz_path_list2):
    if len(all_npz_path_list1) != len(all_npz_path_list2):
        print("提取的分段特征数量不同")
        return False
    list_len = len(all_npz_path_list1)
    all_not_same_npz_file_name_list = []
    all_same_split_feature_name_list = []
    all_not_same_split_feature_name_list = []
    for i in range(list_len):
        not_same_npz_file_name_list, same_split_feature_name_list, not_same_split_feature_name_list = \
            compare_feature_numpy(all_npz_path_list1[i], all_npz_path_list2[i])
        all_not_same_npz_file_name_list.extend(not_same_npz_file_name_list)
        all_same_split_feature_name_list.extend(same_split_feature_name_list)
        all_not_same_split_feature_name_list.extend(not_same_split_feature_name_list)

    return all_not_same_npz_file_name_list, all_same_split_feature_name_list, all_not_same_split_feature_name_list


def check_all_feature(all_npz_root_path1, all_npz_root_path2):
    # all_npz_root_path1 = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/"
    # all_npz_root_path2 = "D:/PycharmProjects/ME-GCN-Project/feature_segment_apex/train"
    all_npz_path_list1 = get_all_npz_path_list(all_npz_root_path1)
    all_npz_path_list2 = get_all_npz_path_list(all_npz_root_path2)
    all_not_same_npz_file_name_list, all_same_split_feature_name_list, all_not_same_split_feature_name_list = \
        compare_all_feature_numpy(all_npz_path_list1, all_npz_path_list2)
    if len(all_not_same_npz_file_name_list):
        print("不是同一分段特征", f"总共{len(all_not_same_npz_file_name_list)}对")
        for item in all_not_same_npz_file_name_list:
            print(item[0], item[1])
    print("-----------------------------------------------------------------------------------------------------------")
    if len(all_same_split_feature_name_list):
        print("分段特征 内容相同", f"总共{len(all_same_split_feature_name_list)}个")
        for item in all_same_split_feature_name_list:
            print(item)
    print("-----------------------------------------------------------------------------------------------------------")
    if len(all_not_same_split_feature_name_list):
        print("分段特征 内容不完全相同", f"总共{len(all_not_same_split_feature_name_list)}个")
        for item in all_not_same_split_feature_name_list:
            print(item)


if __name__ == "__main__":
    all_npz_root_path1 = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/"
    all_npz_root_path2 = "D:/PycharmProjects/ME-GCN-Project/feature_segment_apex/train"
    check_all_feature(all_npz_root_path1, all_npz_root_path2)
    check_all_label(all_npz_root_path1, all_npz_root_path2)
