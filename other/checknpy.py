import numpy as np

# 读取 npy 文件
file_path = 'D:/PycharmProjects/ME-GCN-Project/assets/cas(me)^2.npy'  # 替换为你的 .npy 文件路径
data = np.load(file_path)

# 打印文件内容
print(data)

# 打印数据的形状和类型
print(f'Shape: {data.shape}')
print(f'Data type: {data.dtype}')
