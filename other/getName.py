import re

# 给定的路径信息
text = """
/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s37/happy3_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_6/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_5/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/anger1_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_8/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_6/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_9/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_7/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_4/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/disgust2_5/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_3/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy3_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy2_2/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_4/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/happy1_7/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s27/anger1_1/landmarks.csv does not exist
/kaggle/working/data/cas(me)^2/cropped_apex/s24/happy4_4/landmarks.csv does not exist
"""

# 将路径提取为列表
lines = text.splitlines()
path_list = [line.split()[0] for line in lines if line]

# 打印结果
print(path_list)

# 存储结果的数组
extracted_strings = []

# 使用正则表达式提取 s27 到 landmarks 之间的内容
pattern = re.compile(r"/s27/(.*)/landmarks\.csv")

for path in path_list:
    match = pattern.search(path)
    if match:
        extracted_strings.append(match.group(1))

# 打印结果
print(extracted_strings)
