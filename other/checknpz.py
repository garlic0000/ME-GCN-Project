import numpy as np

# 加载 .npz 文件
data = np.load('D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/casme_015/casme_015_0101_0000.npz')

# 查看文件中存储的数组名称
print(data.files)

# # 读取并打印某个数组的内容（例如数组名称为 'arr_0'）
print(data['feature'])
