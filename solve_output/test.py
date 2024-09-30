import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib.lines as mlines
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体


def visualize_and_save_output(df, video_name, save_output_root_path):
    # 按起始帧 升序排列
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
    save_path = os.path.join(save_output_root_path, f"{video_name}_表情识别结果.png")
    plt.savefig(save_path, bbox_inches='tight')

    # 显示图形
    plt.show()
    # 关闭图形
    plt.close()


if __name__ == "__main__":
    # 创建数据框
    data = {
        'start_frame': [394, 88, 88, 123, 1346, 123],
        'end_frame': [418, 105, 144, 148, 1394, 128],
        'score': [0.275607325, 0.261960912, 0.180311449, 0.056405561, 0.052709335, 0.005885339],
        'type_idx': [1, 1, 1, 1, 1, 2]
    }

    df = pd.DataFrame(data)
    save_output_root_path = "D:/PycharmProjects/ME-GCN-Project/solve_output"
    visualize_and_save_output(df, "test", save_output_root_path)
