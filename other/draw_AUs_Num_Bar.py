import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
# 绘制AUs 柱状图

all_data_dict = {'1': 47, '2': 84, '4': 198, '5': 55, '6': 85, '7': 155, '9': 19, '10': 41,
                 '11': 6, '12': 261, '14': 121, '15': 25, '17': 38, '18': 21, '19': 6, '20': 18,
                 '21': 3, '23': 14, '24': 87, '25': 49, '28': 1, '30': 8, '34': 1, '38': 20,
                 '39': 14, '42': 2, '43': 38, '45': 6, '62': 8, '3': 1, '8': 6, '13': 6,
                 '16': 1, '26': 36, '27': 1, '31': 12, '33': 1, '37': 1, '50': 1, '51': 1,
                 '52': 1, '53': 1, '54': 5, '55': 2, '56': 4, '58': 2, '61': 8, '63': 1,
                 '64': 2, '80': 4, '81': 2, '85': 2}

# 将字典按值降序排序
sorted_items = sorted(all_data_dict.items(), key=lambda x: x[1], reverse=True)

# 提取前30个（如果字典中项少于30个，就提取所有项）
top_30_items = sorted_items[:30]

# 解包数据
keys, values = zip(*top_30_items)

# 绘制柱状图
plt.figure(figsize=(12, 8))
plt.bar(keys, values, color='#3D64AC')
plt.xlabel('动作单元 (AU)', fontsize=20)
plt.ylabel('数量(个)', fontsize=20)
plt.xticks(rotation=45, fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.show()
