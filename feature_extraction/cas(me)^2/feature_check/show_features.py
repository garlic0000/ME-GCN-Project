import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import glob
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体


def load_npz_file(npz_file_path):
    data = np.load(npz_file_path)
    # 输出data的组成
    # print(data.files)
    # ['feature', 'label', 'video_name']
    feature = data['feature']  # 提取特征矩阵
    label = data['label']  # 提取标签矩阵
    video_name = data['video_name']  # 提取视频名称
    return feature, label, video_name


def visualize_feature_at_frame(feature, frame_index):
    frame_data = feature[frame_index]  # 选择某一帧的特征
    print(frame_data)
    x_coords = frame_data[:, 0]  # x 坐标
    y_coords = frame_data[:, 1]  # y 坐标

    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, c='blue', marker='o')
    plt.title(f"Feature points at frame {frame_index}")
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")
    plt.gca().invert_yaxis()  # 翻转y轴以符合图像坐标系
    plt.show()


if __name__ == "__main__":
    all_npz_root_path = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/"
    npz_file_path1 = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/casme_015/casme_015_0101_0000.npz"
    feature, label, video_name = load_npz_file(npz_file_path1)
    print(feature.shape)  # (270, 12, 2)
    print(label.shape)  # (270, 8)
    print(video_name.shape)
    # 可视化第100帧的特征
    visualize_feature_at_frame(feature, frame_index=100)

    npz_file_path2 = "D:/PycharmProjects/ME-GCN-Project/feature_segment_apex_1/train/casme_015/casme_015_0101_0000.npz"
    feature, label, video_name = load_npz_file(npz_file_path2)
    # 可视化第100帧的特征
    visualize_feature_at_frame(feature, frame_index=100)
