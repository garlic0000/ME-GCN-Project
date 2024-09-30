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
    feature = data['feature']  # 提取特征矩阵
    label = data['label']  # 提取标签矩阵
    video_name = data['video_name']  # 提取视频名称
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


def visualize_and_save_micro_label(npz_file_name, label, save_root_path):
    # 创建时间轴
    time_axis = np.arange(label.shape[0])

    # 定义红色作为微表情的颜色
    micro_color = 'red'

    # 微表情标记 (红色)
    micro_start_indices = time_axis[label[:, 0] == 1]
    micro_apex_indices = time_axis[label[:, 1] == 1]
    micro_end_indices = time_axis[label[:, 2] == 1]
    micro_action_indices = time_axis[label[:, 3] == 1]

    if len(micro_action_indices) > 0:
        # 可视化每个标签随时间的变化
        plt.figure(figsize=(12, 6))
        # 绘制微表情
        plt.plot(micro_start_indices, [1] * len(micro_start_indices), '>', label="微表情 起始帧",
                 color=micro_color)  # 起始帧
        plt.plot(micro_apex_indices, [1] * len(micro_apex_indices), 'o', label="微表情 顶点帧",
                 color=micro_color)  # 顶点帧
        plt.plot(micro_end_indices, [1] * len(micro_end_indices), '<', label="微表情 结束帧", color=micro_color)  # 结束帧
        plt.plot(micro_action_indices, [1] * len(micro_action_indices), '-', label="微表情 表情帧区间",
                 color=micro_color)  # 动作阶段

        # 标记起始、顶点和结束帧的数值
        for idx in micro_start_indices:
            plt.text(idx, 1.05, f'{idx}', color=micro_color, ha='center', fontsize=10)
        for idx in micro_apex_indices:
            plt.text(idx, 1.05, f'{idx}', color=micro_color, ha='center', fontsize=10)
        for idx in micro_end_indices:
            plt.text(idx, 1.05, f'{idx}', color=micro_color, ha='center', fontsize=10)
        # 设置 y 轴刻度显示1
        plt.yticks([1])

        # 设置 y 轴范围
        plt.ylim(0.5, 1.2)

        # 设置 x 轴范围
        plt.xlim(0, 270)

        # 设置 x 轴刻度每隔 27 显示一个刻度
        plt.xticks(np.arange(0, 271, 27))  # 从 0 到 270，每隔 27 标一个刻度

        # 添加标题和标签
        plt.title(f"{npz_file_name} 微表情标签")
        plt.xlabel("帧")
        plt.ylabel("标签值")

        # 将图例移动到正中间
        # 不能压线
        plt.legend(loc='center', bbox_to_anchor=(0.5, 0.2), frameon=False)

        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)
        # 保存图像为 PNG 文件
        save_path = os.path.join(save_root_path, f"{npz_file_name}.png")
        if os.path.exists(save_path):
            os.remove(save_path)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存图像，设置分辨率为300 DPI

        # # 显示图像
        # plt.show()

        # 关闭当前图像 释放内存
        plt.close()


def visualize_and_save_macro_label(npz_file_name, label, save_root_path):
    # 创建时间轴
    time_axis = np.arange(label.shape[0])

    # 定义蓝色作为宏表情的颜色
    macro_color = 'blue'


    # 宏表情标记 (蓝色)
    macro_start_indices = time_axis[label[:, 4] == 1]
    macro_apex_indices = time_axis[label[:, 5] == 1]
    macro_end_indices = time_axis[label[:, 6] == 1]
    macro_action_indices = time_axis[label[:, 7] == 1]

    if len(macro_action_indices) > 0:
        # 可视化每个标签随时间的变化
        plt.figure(figsize=(12, 6))
        # 绘制宏表情
        plt.plot(macro_start_indices, [1] * len(macro_start_indices), '>', label="宏表情 起始帧",
                 color=macro_color)  # 起始帧
        plt.plot(macro_apex_indices, [1] * len(macro_apex_indices), 'x', label="宏表情 顶点帧",
                 color=macro_color)  # 顶点帧
        plt.plot(macro_end_indices, [1] * len(macro_end_indices), '<', label="宏表情 结束帧", color=macro_color)  # 结束帧
        plt.plot(macro_action_indices, [1] * len(macro_action_indices), '-', label="宏表情 表情帧区间",
                 color=macro_color)  # 动作阶段

        for idx in macro_start_indices:
            plt.text(idx, 1.05, f'{idx}', color=macro_color, ha='center', fontsize=10)
        for idx in macro_apex_indices:
            plt.text(idx, 1.05, f'{idx}', color=macro_color, ha='center', fontsize=10)
        for idx in macro_end_indices:
            plt.text(idx, 1.05, f'{idx}', color=macro_color, ha='center', fontsize=10)

        # 设置 y 轴刻度显示1
        plt.yticks([1])

        # 设置 y 轴范围
        plt.ylim(0.5, 1.2)

        # 设置 x 轴范围
        plt.xlim(0, 270)

        # 设置 x 轴刻度每隔 27 显示一个刻度
        plt.xticks(np.arange(0, 271, 27))  # 从 0 到 270，每隔 27 标一个刻度

        # 添加标题和标签
        plt.title(f"{npz_file_name} 宏表情标签")
        plt.xlabel("帧")
        plt.ylabel("标签值")

        # 将图例移动到正中间
        # 不能压线
        plt.legend(loc='center', bbox_to_anchor=(0.5, 0.2), frameon=False)

        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)
        # 保存图像为 PNG 文件
        save_path = os.path.join(save_root_path, f"{npz_file_name}.png")
        if os.path.exists(save_path):
            os.remove(save_path)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)  # 保存图像，设置分辨率为300 DPI

        # # 显示图像
        # plt.show()

        # 关闭当前图像 释放内存
        plt.close()


def get_npz_count(all_npz_root_path):
    # 获取所有npz文件的个数
    count = 0
    for sub_item in Path(all_npz_root_path).iterdir():
        if not sub_item.is_dir():
            continue
            # # 计算目录下所有 .jpg 文件的数量
        count += len(glob.glob(os.path.join(str(sub_item), "*.npz")))
    return count


def show_all_labels(all_npz_root_path, save_micro_root_path, save_macro_root_path):
    sum_count = get_npz_count(all_npz_root_path)
    print(f"npz文件的个数为{sum_count}")
    with tqdm(total=sum_count) as tq:
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
                visualize_and_save_micro_label(npz_file_name, label, save_micro_root_path)
                visualize_and_save_macro_label(npz_file_name, label, save_macro_root_path)
                tq.update()


if __name__ == "__main__":
    all_npz_root_path = "D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/"
    save_micro_root_path = "D:/PycharmProjects\ME-GCN-Project/feature_extraction/cas(me)^2/feature_check/micro_labelpic/"
    save_macro_root_path = "D:/PycharmProjects\ME-GCN-Project/feature_extraction/cas(me)^2/feature_check/macro_labelpic/"
    show_all_labels(all_npz_root_path, save_micro_root_path, save_macro_root_path)
    # # 可视化第100帧的特征
    # visualize_feature_at_frame(feature, frame_index=100)
    # # 示例标签矩阵
    # label = np.random.randint(0, 2, size=(270, 8))  # 假设有8个标签，100个时间点

    # 可视化标签分布
