import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps


def load_npz_file(npz_file_path):
    data = np.load(npz_file_path)
    feature = data['feature']  # 提取特征矩阵
    label = data['label']  # 提取标签矩阵
    video_name = data['video_name']  # 提取视频名称
    print("Feature shape:", feature.shape)
    print("Label shape:", label.shape)
    print("Video Name:", video_name)
    return feature, label, video_name


def visualize_feature_at_frame(feature, frame_index):
    frame_data = feature[frame_index]  # 选择某一帧的特征
    x_coords = frame_data[:, 0]  # x 坐标
    y_coords = frame_data[:, 1]  # y 坐标

    plt.figure(figsize=(6, 6))
    plt.scatter(x_coords, y_coords, c='blue', marker='o')
    plt.title(f"Feature points at frame {frame_index}")
    plt.xlabel("X coordinates")
    plt.ylabel("Y coordinates")
    plt.gca().invert_yaxis()  # 翻转y轴以符合图像坐标系
    plt.show()


def visualize_label_distribution_with_markers(label):
    # 创建时间轴
    time_axis = np.arange(label.shape[0])

    # 定义红色和蓝色作为微表情和宏表情的颜色
    micro_color = 'red'
    macro_color = 'blue'

    # 可视化每个标签随时间的变化
    plt.figure(figsize=(12, 6))

    # 微表情标记 (红色)
    micro_start_indices = time_axis[label[:, 0] == 1]
    micro_apex_indices = time_axis[label[:, 1] == 1]
    micro_end_indices = time_axis[label[:, 2] == 1]
    micro_action_indices = time_axis[label[:, 3] == 1]

    plt.plot(micro_start_indices, [1] * len(micro_start_indices), 'o', label="Micro Start", color=micro_color)  # 起始帧
    plt.plot(micro_apex_indices, [1] * len(micro_apex_indices), 'o', label="Micro Apex", color=micro_color)  # 顶点帧
    plt.plot(micro_end_indices, [1] * len(micro_end_indices), 'o', label="Micro End", color=micro_color)  # 结束帧
    plt.plot(micro_action_indices, [1] * len(micro_action_indices), '-', label="Micro Action",
             color=micro_color)  # 动作阶段

    # 宏表情标记 (蓝色)
    macro_start_indices = time_axis[label[:, 4] == 1]
    macro_apex_indices = time_axis[label[:, 5] == 1]
    macro_end_indices = time_axis[label[:, 6] == 1]
    macro_action_indices = time_axis[label[:, 7] == 1]

    plt.plot(macro_start_indices, [1] * len(macro_start_indices), 'x', label="Macro Start", color=macro_color)  # 起始帧
    plt.plot(macro_apex_indices, [1] * len(macro_apex_indices), 'x', label="Macro Apex", color=macro_color)  # 顶点帧
    plt.plot(macro_end_indices, [1] * len(macro_end_indices), 'x', label="Macro End", color=macro_color)  # 结束帧
    plt.plot(macro_action_indices, [1] * len(macro_action_indices), '-', label="Macro Action",
             color=macro_color)  # 动作阶段

    # 标记起始、顶点和结束帧的数值
    for idx in micro_start_indices:
        plt.text(idx, 1.05, f'{idx}', color=micro_color, ha='center', fontsize=10)
    for idx in micro_apex_indices:
        plt.text(idx, 1.05, f'{idx}', color=micro_color, ha='center', fontsize=10)
    for idx in micro_end_indices:
        plt.text(idx, 1.05, f'{idx}', color=micro_color, ha='center', fontsize=10)

    for idx in macro_start_indices:
        plt.text(idx, 1.05, f'{idx}', color=macro_color, ha='center', fontsize=10)
    for idx in macro_apex_indices:
        plt.text(idx, 1.05, f'{idx}', color=macro_color, ha='center', fontsize=10)
    for idx in macro_end_indices:
        plt.text(idx, 1.05, f'{idx}', color=macro_color, ha='center', fontsize=10)

    # 设置 y 轴刻度只显示 0 和 1
    plt.yticks([0, 1])

    # 设置 y 轴范围
    plt.ylim(-0.1, 1.2)

    # 添加标题和标签
    plt.title("Label Distribution Over Time (Only 1s)")
    plt.xlabel("Time")
    plt.ylabel("Label Value (1s Only)")

    # 将图例移动到正中间
    # 不能压线
    plt.legend(loc='center', bbox_to_anchor=(0.5, 0.2), frameon=False)

    # 显示图像
    plt.show()


if __name__ == "__main__":
    npz_file_path = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/casme_015/casme_015_0101_1024.npz"  # 替换为实际的路径
    feature, label, video_name = load_npz_file(npz_file_path)
    # 可视化第100帧的特征
    visualize_feature_at_frame(feature, frame_index=100)
    # # 示例标签矩阵
    # label = np.random.randint(0, 2, size=(270, 8))  # 假设有8个标签，100个时间点

    # 可视化标签分布
    visualize_label_distribution_with_markers(label)
