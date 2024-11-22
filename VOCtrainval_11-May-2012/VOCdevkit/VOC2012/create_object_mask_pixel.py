import os
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

def create_mask(xml_path, output_dir):
    # 解析XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取图像大小
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    # 创建空白掩膜
    mask = np.zeros((height, width), dtype=np.uint8)
    current_label = 1  # 起始标签值

    # 解析每个边界框的坐标
    objects = root.findall('object')
    for obj in objects:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 将边界框区域设置为对应的标签值
        mask[ymin:ymax, xmin:xmax] = current_label
        current_label += 1

    # 创建掩膜图像
    mask_img = Image.fromarray(mask)

    # 保存掩膜图像
    mask_filename = os.path.splitext(os.path.basename(xml_path))[0] + '.png'
    output_path = os.path.join(output_dir, mask_filename)
    mask_img.save(output_path)

    print("生成掩膜完成！")



# 目录路径
directory = 'E:/deep-learning-for-image-processing-pytorch_object_detection/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations'

# 获取目录下所有xml文件
file_list = [filename for filename in os.listdir(directory) if filename.endswith('.xml')]

# 指定XML文件路径和输出目录
for filename in file_list:
    xml_path = os.path.join(directory, filename)
    output_dir = 'SegmentationObject'
    create_mask(xml_path, output_dir)  