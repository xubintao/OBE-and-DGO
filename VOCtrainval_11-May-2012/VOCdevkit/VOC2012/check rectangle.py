import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw

def draw_boxes(image_path, annotation_path, output_path):
    # 加载图片
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # 解析XML文件
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # 遍历所有的object元素
    for obj in root.findall('object'):
        # 获取bounding box的坐标
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # 绘制矩形框
        draw.rectangle((xmin, ymin, xmax, ymax), outline='red', width=3)

    # 保存图片
    image.save(output_path)

def process_folder(image_folder, annotation_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历图片文件夹中的所有文件
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            # 构建完整的文件路径
            image_path = os.path.join(image_folder, filename)
            annotation_filename = filename.split('.')[0] + '.xml'
            annotation_path = os.path.join(annotation_folder, annotation_filename)

            # 检查XML文件是否存在
            if os.path.exists(annotation_path):
                # 构建输出文件的路径
                output_filename = filename
                output_path = os.path.join(output_folder, output_filename)

                # 绘制标注框并保存图片
                draw_boxes(image_path, annotation_path, output_path)
                print(f"Processed {filename}")
            else:
                print(f"Annotation file not found for {filename}")

# 使用示例
image_folder = 'C:/Users/86131/Desktop/train'
annotation_folder = 'C:/Users/86131/Desktop/train'
output_folder = 'C:/Users/86131/Desktop/new'
process_folder(image_folder, annotation_folder, output_folder)