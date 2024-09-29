import os
import glob
import csv
from pathlib import Path

from tqdm import tqdm
import numpy as np
import cv2
import yaml

from tools import get_micro_expression_average_len, calculate_roi_freature_list


def get_flow_count(root_path):
    count = 0
    for sub_item in Path(root_path).iterdir():
        if sub_item.is_dir():
            for type_item in sub_item.iterdir():
                if type_item.is_dir():
                    count += len(glob.glob(os.path.join(
                        str(type_item), "flow_x*.jpg")))
    return count


def feature(opt):
    optflow_root_path = opt["optflow_root_path"]
    feature_root_path = opt["feature_root_path"]
    landmark_root_path = opt["cropped_root_path"]
    anno_csv_path = opt["anno_csv_path"]
    print(f'dataset: {opt["dataset"]}')
    sum_count = get_flow_count(optflow_root_path)
    print("flow count = ", sum_count)

    # opt_step = 1  # int(get_micro_expression_average_len(anno_csv_path) // 2)
    opt_step = int(get_micro_expression_average_len(anno_csv_path) // 2)
    print(f"opt_step: {opt_step}")

    # for debug use
    # short_video_list = []
    with tqdm(total=sum_count) as tq:
        for sub_item in Path(optflow_root_path).iterdir():
            if not sub_item.is_dir():
                continue
            for type_item in sub_item.iterdir():
                if not type_item.is_dir():
                    continue
                # 获取x方向光流文件
                flow_x_path_list = glob.glob(
                    os.path.join(str(type_item), "flow_x*.jpg"))
                # 获取y方向光流文件
                flow_y_path_list = glob.glob(
                    os.path.join(str(type_item), "flow_y*.jpg"))
                flow_x_path_list.sort()
                flow_y_path_list.sort()

                # 关键点文件路径
                csv_landmark_path = os.path.join(
                    landmark_root_path,
                    sub_item.name, type_item.name, "landmarks.csv")
                if not os.path.exists(csv_landmark_path):
                    print("\n")
                    print(f"{csv_landmark_path} does not exist")
                    continue
                with open(csv_landmark_path, 'r') as f:
                    ior_feature_list_sequence = []  # 存储整段视频的ROI特征序列
                    csv_r = list(csv.reader(f))
                    # # 用于调试
                    # row_count = 0
                    # for row in csv_r:
                    #     # 遍历 csv_r 后，文件指针已经到达文件末尾
                    #     # 需要重置 否则会出问题
                    #     # 将 csv_r = csv.reader(f) 转换为list使用
                    #     row_count += 1
                    # print(len(csv_r) == row_count)
                    # print(row_count)
                    # TypeError: object of type '_csv.reader' has no len()
                    # # 用于测试
                    # row_count = len(csv_r)
                    # if row_count > len(flow_x_path_list):
                    #     print("row_count > flow_x_path_list")
                    #     print("video_name:", str(type_item))
                    #     print("row_count:", row_count)
                    #     print("flow num:", len(flow_x_path_list))
                    #     continue

                    # 遍历每一帧的的关键点
                    for index, row in enumerate(csv_r):
                        # 帧的索引小于步长 跳过
                        # 每opt_step帧进行一次光流处理
                        if index < opt_step:
                            # 用于测试
                            # print("index < opt_step")
                            # print(index, opt_step, row)
                            continue
                        i = index - opt_step
                        # 这段有问题
                        flow_x = cv2.imread(flow_x_path_list[i],
                                            cv2.IMREAD_GRAYSCALE)
                        flow_y = cv2.imread(flow_y_path_list[i],
                                            cv2.IMREAD_GRAYSCALE)
                        # 将x方向和y方向的光流堆叠在一起 形成光流特征图
                        flow_x_y = np.stack((flow_x, flow_y), axis=2)
                        flow_x_y = flow_x_y / np.float32(255)  # 归一化
                        # 平移 使光流特征值范围为[-0.5, 0.5]
                        flow_x_y = flow_x_y - 0.5
                        # 将每一帧的68个关键点转换为numpy数组
                        landmarks = np.array(
                            [(int(row[index]), int(row[index + 68]))
                             for index in range(int(len(row) // 2))])

                        try:
                            # 计算该帧的ROI特征序列
                            ior_feature_list = calculate_roi_freature_list(
                                flow_x_y, landmarks, radius=5)
                            # 将该帧的ROI特征堆叠并添加到序列中
                            ior_feature_list_sequence.append(
                                np.stack(ior_feature_list, axis=0))
                            tq.update()
                        except Exception as exp:
                            # 出现错误 清空特征
                            ior_feature_list_sequence = []
                            print("ior_feature_list 有问题")
                            print(f"{sub_item.name}  {type_item.name}")
                            # 打印异常信息
                            print(str(exp))
                            break

                    # 序列中有特征
                    if len(ior_feature_list_sequence) > 0:
                        new_type_dir_path = os.path.join(
                            feature_root_path, sub_item.name)
                        if not os.path.exists(new_type_dir_path):
                            os.makedirs(new_type_dir_path)
                        # 保存为npy文件
                        np.save(os.path.join(
                            new_type_dir_path, f"{type_item.name}.npy"),
                            np.stack(ior_feature_list_sequence, axis=0))

    # print("len of wrong video_list: {}".format(len(short_video_list)))
    # print("*" * 10, "wrong video list", "*" * 10)
    # print(short_video_list)


if __name__ == "__main__":
    with open("/kaggle/working/ME-GCN-Project/feature_extraction/cas(me)^2/config.yaml", encoding="UTF-8") as f:
        yaml_config = yaml.safe_load(f)
        dataset = yaml_config['dataset']
        opt = yaml_config[dataset]

    feature(opt)
