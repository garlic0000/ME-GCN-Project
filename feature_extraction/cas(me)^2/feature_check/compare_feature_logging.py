import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
from pathlib import Path
import glob
import logging


def load_npz_file(npz_file_path):
    data = np.load(npz_file_path)
    feature = data['feature']  # 提取特征矩阵
    label = data['label']  # 提取标签矩阵
    video_name = data['video_name']  # 提取视频名称
    return feature, label, video_name


def get_all_npz_path_list(all_npz_root_path):
    """
    获取所有的npz文件的路径列表
    """
    all_npz_root_path_list = []
    for sub_item in Path(all_npz_root_path).iterdir():
        if not sub_item.is_dir():
            continue
        npz_file_path_list = glob.glob(os.path.join(str(sub_item), "*.npz"))
        npz_file_path_list.sort()
        all_npz_root_path_list.extend(npz_file_path_list)
    all_npz_root_path_list.sort()
    return all_npz_root_path_list


def compare_feature_at_frame(feature1, feature2):
    """
    对分段特征中的每一帧进行比较
    """
    # 对每个分段特征中的每一帧进行对比
    if feature1.shape != feature2.shape:
        logging.info("特征长度不相等")
        return
    min1, max1 = np.min(feature1), np.max(feature1)
    min2, max2 = np.min(feature2), np.max(feature2)
    # # 输出数组的范围
    # logging.info(f"feature1 range: [{min1}, {max1}]")
    # logging.info(f"feature2 range: [{min2}, {max2}]")
    feature_range_list = [(min1, max1), (min2, max2)]
    same_frame_index_list = []  # 相同帧编号
    not_same_frame_index_list = []  # 不同帧编号
    for frame_index in range(0, len(feature1)):
        frame_data1 = feature1[frame_index]  # 选择某一帧的特征
        frame_data2 = feature2[frame_index]
        if np.array_equal(frame_data1, frame_data2):
            same_frame_index_list.append(frame_index)
        else:

            not_same_frame_index_list.append(frame_index)
    return same_frame_index_list, not_same_frame_index_list, feature_range_list


def compare_feature_numpy(npz_file_path1, npz_file_path2):
    """

    """
    feature1, label1, video_name1 = load_npz_file(npz_file_path1)
    feature2, label2, video_name2 = load_npz_file(npz_file_path2)
    npz_file_name1 = os.path.split(npz_file_path1)[-1]
    npz_file_name2 = os.path.split(npz_file_path2)[-1]
    not_same_npz_file_name_list = []
    same_split_feature_name_list = []
    not_same_split_feature_name_list = []
    same_frame_index_dict = {}
    not_same_frame_index_dict = {}
    not_same_feature_range_dict = {}
    if npz_file_name1 != npz_file_name2:
        # logging.info(f"不是同一分段特征")
        not_same_npz_file_name_list.append((npz_file_name1, npz_file_name2))
        return not_same_npz_file_name_list, [], []
    npz_file_name = os.path.splitext(npz_file_name1)[0]
    if np.array_equal(feature1, feature2):
        # 分段特征相同
        same_split_feature_name_list.append(npz_file_name)
    else:
        # 分段特征不同
        not_same_split_feature_name_list.append(npz_file_name)
        # 不同的分段特征 按帧比较
        same_frame_index_list, not_same_frame_index_list, feature_range_list = compare_feature_at_frame(feature1,
                                                                                                        feature2)
        same_frame_index_dict[npz_file_name] = same_frame_index_list
        not_same_frame_index_dict[npz_file_name] = not_same_frame_index_list
        not_same_feature_range_dict[npz_file_name] = feature_range_list
    return not_same_npz_file_name_list, same_split_feature_name_list, not_same_split_feature_name_list, same_frame_index_dict, not_same_frame_index_dict, not_same_feature_range_dict


def compare_all_feature_numpy(all_npz_path_list1, all_npz_path_list2):
    if len(all_npz_path_list1) != len(all_npz_path_list2):
        logging.info("提取的分段特征数量不同")
        return False
    list_len = len(all_npz_path_list1)
    all_not_same_npz_file_name_list = []
    all_same_split_feature_name_list = []
    all_not_same_split_feature_name_list = []
    all_same_frame_index_dict = {}
    all_not_same_frame_index_dict = {}
    all_not_same_feature_range_dict = {}
    for i in range(list_len):
        not_same_npz_file_name_list, same_split_feature_name_list, not_same_split_feature_name_list, same_frame_index_dict, not_same_frame_index_dict, not_same_feature_range_dict = \
            compare_feature_numpy(all_npz_path_list1[i], all_npz_path_list2[i])
        all_not_same_npz_file_name_list.extend(not_same_npz_file_name_list)
        all_same_split_feature_name_list.extend(same_split_feature_name_list)
        all_not_same_split_feature_name_list.extend(not_same_split_feature_name_list)
        all_same_frame_index_dict.update(same_frame_index_dict)
        all_not_same_frame_index_dict.update(not_same_frame_index_dict)
        all_not_same_feature_range_dict.update(not_same_feature_range_dict)
    return all_not_same_npz_file_name_list, all_same_split_feature_name_list, all_not_same_split_feature_name_list, all_same_frame_index_dict, all_not_same_frame_index_dict, all_not_same_feature_range_dict


def check_all_feature(all_npz_root_path1, all_npz_root_path2):
    all_npz_path_list1 = get_all_npz_path_list(all_npz_root_path1)
    all_npz_path_list2 = get_all_npz_path_list(all_npz_root_path2)
    all_not_same_npz_file_name_list, all_same_split_feature_name_list, all_not_same_split_feature_name_list, all_same_frame_index_dict, all_not_same_frame_index_dict, all_not_same_feature_range_dict = \
        compare_all_feature_numpy(all_npz_path_list1, all_npz_path_list2)
    logging.info(f"分段特征 名称不同 总共{len(all_not_same_npz_file_name_list)}对")
    if len(all_not_same_npz_file_name_list):
        for item in all_not_same_npz_file_name_list:
            logging.info(item[0], item[1])
    logging.info(
        "-----------------------------------------------------------------------------------------------------------")
    logging.info(f"分段特征 内容相同 总共{len(all_same_split_feature_name_list)}对")
    if len(all_same_split_feature_name_list):
        for item in all_same_split_feature_name_list:
            logging.info(item)
    logging.info(
        "-----------------------------------------------------------------------------------------------------------")
    logging.info(f"分段特征 内容不完全相同 总共{len(all_not_same_split_feature_name_list)}对")
    logging.info("*********************************************************************************************")
    if len(all_not_same_split_feature_name_list):
        for item in all_not_same_split_feature_name_list:
            logging.info(item)
            # [(,),(,)]
            logging.info(
                f"仓库提供的特征 范围: [{all_not_same_feature_range_dict[item][0][0]}, {all_not_same_feature_range_dict[item][0][1]}]")
            logging.info(
                f"自己提取的特征 范围: [{all_not_same_feature_range_dict[item][1][0]}, {all_not_same_feature_range_dict[item][1][1]}]")
            logging.info("256帧为一段 填充至270帧 (270, 12, 2)")
            logging.info("内容完全相同的帧编号:")
            logging.info(all_same_frame_index_dict[item])
            logging.info("内容不完全相同的帧编号:")
            logging.info(all_not_same_frame_index_dict[item])
            logging.info("*********************************************************************************************")

    logging.info(
        "-----------------------------------------------------------------------------------------------------------")


def setup_logging(log_file_path):
    """
    记录日志
    """
    logging.basicConfig(
        level=logging.INFO,  # 日志级别
        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
        handlers=[
            logging.FileHandler(log_file_path),  # 将日志写入文件
            logging.StreamHandler()  # 同时输出到控制台（可以根据需要移除）
        ]
    )


if __name__ == "__main__":
    # 设置日志文件路径
    log_file_path = "./compare_feature_logfile.log"
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
    # 初始化日志记录
    setup_logging(log_file_path)
    all_npz_root_path1 = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train"
    all_npz_root_path2 = "D:/PycharmProjects/ME-GCN-Project/feature_segment_apex_1/train"
    logging.info("START")
    check_all_feature(all_npz_root_path1, all_npz_root_path2)
    logging.info("END")
