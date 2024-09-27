import numpy as np

# 加载 .npz 文件
data1 = np.load('D:/PycharmProjects/ME-GCN-Project/features/cas(me)^2/feature_segment/train/casme_015/casme_015_0101_0000.npz')

data2 = np.load('D:/PycharmProjects/ME-GCN-Project/feature_segment_apex/train/casme_015/casme_015_0101_0000.npz')

# 查看文件中存储的数组名称
print(data1.files)
print(data2.files)

# # 读取并打印某个数组的内容（例如数组名称为 'arr_0'）
print(data1['feature'])
print("dfhdkjfhdsgfg")
print(data2['feature'])
# 比较两个数组中每个位置的元素
differences = [(i, a, b) for i, (a, b) in enumerate(zip(data1['feature'], data2['feature'])) if not np.array_equal(a, b)]
print(differences)  # 输出 [(1, 2, 3), (2, 3, 2)]，表示位置 1 和 2 的元素不同