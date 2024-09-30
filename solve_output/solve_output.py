import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体


def choose_best_epoch():
    """
    根据指标
    从训练epoch_metrics.csv 中选择表现最好的 epoch 获取epoch编号
    从 nms_csv中获取对应epoch文件

    """
    # 选择指标的范围
    # 默认是 all_f1
    out_root_path = "C:/Users/garlic/Downloads/ckpt/output/casme"
    best_epoch_dict = {}
    for sub_item in Path(out_root_path).iterdir():
        if not sub_item.is_dir():
            continue
        epoch_metrics_file = os.path.join(str(sub_item), 'epoch_metrics.csv')

        if os.path.exists(epoch_metrics_file):
            df = pd.read_csv(epoch_metrics_file)
            # 检查文件是否为空
            if df.empty:
                best_epoch_dict[sub_item.name] = -1
                continue
            # 降序排列 选择指标值 最高的
            df.sort_values('all_f1', inplace=True, ascending=False)
            df = df.reset_index(drop=True)
            # 获取最佳 epoch
            best_epoch = df.loc[0, 'epoch']
            best_epoch_dict[sub_item.name] = int(best_epoch)  # 确保 epoch 是整数类型
        else:
            # 没有epoch_metrics.csv 则填 空
            best_epoch_dict[sub_item.name] = -1

    return best_epoch_dict


def get_nms_csv_path_dict(best_epoch_dict):
    out_root_path = "C:/Users/garlic/Downloads/ckpt/output/casme"
    nms_csv_path_dict = {}
    for sub_item in Path(out_root_path).iterdir():
        if not sub_item.is_dir():
            continue
        nms_csv_path = os.path.join(str(sub_item), 'nms_csv',
                                    f'final_proposals_epoch_0{best_epoch_dict[sub_item.name]}.csv')
        if not os.path.exists(nms_csv_path):
            nms_csv_path_dict[sub_item.name] = ""
        nms_csv_path_dict[sub_item.name] = nms_csv_path
    return nms_csv_path_dict


def visualize_and_save_output(df, video_name, save_output_root_path):
    # 按起始帧 升序排列
    df = df.copy()  # 创建一个 DataFrame 的副本
    df.sort_values('start_frame', inplace=True, ascending=True)
    df = df.reset_index(drop=True)
    # 遍历每一行数据进行绘制
    for i, row in df.iterrows():
        start_frame = row['start_frame']
        end_frame = row['end_frame']
        score = row['score']
        type_idx = row['type_idx']

        # 确定颜色，type_idx 为 1 的用蓝色，为 2 的用红色
        color = 'blue' if type_idx == 1 else 'red'

        # 画线段，左端是 start_frame，右端是 end_frame，线段的高度是 i+1，表示第几行
        plt.plot([start_frame, end_frame], [i + 1, i + 1], color=color, lw=2)

        # 在线段的左端绘制 start_frame 值，右端绘制 end_frame 值
        plt.text(start_frame - 10, i + 1, f"{int(start_frame)}", va='center', ha='right', fontsize=8, color=color)
        plt.text(end_frame + 10, i + 1, f"{int(end_frame)}", va='center', ha='left', fontsize=8, color=color)

        # 在线段的上端绘制 score 值
        plt.text((start_frame + end_frame) / 2, i + 1 + 0.1, f"{score:.4f}", va='bottom', ha='center', fontsize=8,
                 color=color)
    # 设置 x 轴和 y 轴的标签
    plt.xlabel('帧')
    plt.ylabel('表情序号')

    # y 轴刻度设置，根据 df 的行数确定刻度
    plt.yticks(range(1, len(df) + 1))

    # 设置 x 轴范围，确保显示完全
    plt.xlim(0, max(df['end_frame']) + 100)

    # 设置图形标题
    plt.title(f"{video_name} 表情识别结果")
    # 添加图例说明，蓝色线段表示宏表情，红色线段表示微表情
    blue_line = mlines.Line2D([], [], color='blue', lw=2, label='宏表情')
    red_line = mlines.Line2D([], [], color='red', lw=2, label='微表情')
    plt.legend(handles=[blue_line, red_line], loc='lower right')
    # 保存图形
    if not os.path.exists(save_output_root_path):
        os.makedirs(save_output_root_path)
    save_path = os.path.join(save_output_root_path, f"{video_name}_表情识别结果.png")
    if os.path.exists(save_path):
        os.remove(save_path)
    plt.savefig(save_path, bbox_inches='tight')

    # # 显示图形
    # plt.show()
    # 关闭图形
    plt.close()


def visualize_and_save_all_output(nms_csv_path_dict, save_output_root_path):
    for nms_csv_path in nms_csv_path_dict.values():
        if nms_csv_path != "" and os.path.exists(nms_csv_path):
            df = pd.read_csv(nms_csv_path)
            # 检查文件是否为空
            if df.empty:
                continue
            # 检查是否存在 video_name 列
            if "video_name" not in df.columns:
                raise ValueError("The file does not contain a 'video_name' column.")

            # 获取去重的 video_name 列
            unique_video_names = df["video_name"].unique()
            video_names_list = unique_video_names.tolist()
            for video_name in video_names_list:
                df_u_v = df[df["video_name"] == video_name]
                visualize_and_save_output(df_u_v, video_name, save_output_root_path)


def solve_output():
    save_output_root_path = "D:/PycharmProjects/ME-GCN-Project/solve_output/visualize_output"
    best_epoch_dict = choose_best_epoch()
    print(best_epoch_dict)
    nms_csv_path_dict = get_nms_csv_path_dict(best_epoch_dict)
    print(nms_csv_path_dict)
    visualize_and_save_all_output(nms_csv_path_dict, save_output_root_path)


if __name__ == "__main__":
    solve_output()
