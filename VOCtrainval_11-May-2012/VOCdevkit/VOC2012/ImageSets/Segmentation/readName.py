import os
import random

# 目录路径
directory = 'E:/deep-learning-for-image-processing-pytorch_object_detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/input'

# 获取目录下所有jpg文件
file_list = [filename for filename in os.listdir(directory) if filename.endswith('.jpg')]

# 随机打乱文件列表
random.shuffle(file_list)

# 按照比例划分文件名
train_ratio = 1  # 训练集比例
train_size = int(len(file_list) * train_ratio)
train_files = file_list[:train_size]
val_files = file_list[train_size:]

# 写入train.txt
with open('train.txt', 'w') as file:
    for filename in train_files:
        file.write(os.path.splitext(filename)[0] + '\n')

# # 写入val.txt
# with open('val.txt', 'w') as file:
#     for filename in val_files:
#         file.write(os.path.splitext(filename)[0] + '\n')
