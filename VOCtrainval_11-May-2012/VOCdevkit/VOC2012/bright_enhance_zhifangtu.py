import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET


"""
在threshold=128的情况下，增强亮度1.1倍效果最好，现在我们要在1.1倍的情况下找到threshold最好的情况
目前采用的计算threshold的方法是直方图分析计算中位数

请注意，multiplier 和 threshold 是可以调整的参数，您可能需要根据您的具体图像数据进行实验来找到最佳的设置。
此外，直方图均衡化可能会影响整个图像的对比度，而不仅仅是 bbox 区域，因此请仔细调整以获得最佳效果。
"""
def parse_xml(xml_file, img_size):
    """
    解析PASCAL VOC格式的XML文件，返回边界框列表和区域的标签。
    
    :param xml_file: XML标注文件路径
    :param img_size: 图像的尺寸，用于处理不同尺寸的图像
    :return: 边界框列表和对应的标签列表
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    bboxes = []
    labels = []
    for obj in root.iter('object'):
        bbox = obj.find('bndbox')
        x = int(float(bbox.find('xmin').text))
        y = int(float(bbox.find('ymin').text))
        w = int(float(bbox.find('xmax').text)) - x
        h = int(float(bbox.find('ymax').text)) - y
        bboxes.append((x, y, w, h))
        labels.append(obj.find('name').text)
    
    # 确保bbox在图像尺寸范围内
    bboxes = [(bbox[0], bbox[1], min(bbox[2], img_size[0] - bbox[0]), min(bbox[3], img_size[1] - bbox[1])) for bbox in bboxes]
    
    return bboxes, labels

def calculate_threshold_from_histogram(roi):
    # 计算直方图
    hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    
    # 找到直方图的中位数索引
    total_pixels = np.sum(hist)
    cumsum = np.cumsum(hist)
    threshold = np.where(cumsum >= total_pixels / 2)[0][0]
    
    return threshold

def enhance_brightness(image, bbox, multiplier=1.1):
    """
    对给定图像中的特定区域（bbox）进行亮度增强，同时尽量减少对整体图像对比度的影响。
    
    :param image: 输入图像
    :param bbox: 目标区域的边界框，格式为[x, y, width, height]
    :param multiplier: 亮度增强的乘数
    :param threshold: 亮度阈值，只有高于该阈值的像素会被增强
    :return: 增强后的图像
    """
    x, y, w, h = bbox
    roi = image[y:y+h, x:x+w]
    original_roi = roi.copy()  # 保存原始 ROI 以保留对比度
    # print(roi)
    # 计算阈值
    threshold = calculate_threshold_from_histogram(roi)
    print(threshold)
    
    roi = roi.astype(np.float32)

    # 先归一化，再增强高于阈值的像素
    float_roi=roi/255
    enhanced_roi = np.where(float_roi > threshold/255, np.minimum(float_roi * multiplier, 1), float_roi)

    # 确保像素值在 0-255 的范围内，并转化为整数
    enhanced_roi=enhanced_roi*255
    enhanced_roi = np.clip(enhanced_roi, 0, 255).astype('uint8')
    # print(enhanced_roi)

    # 将增强的区域放回原图像
    image[y:y+h, x:x+w] = enhanced_roi
    
    return image

def process_images_and_annotations(folder_path, output_folder):
    """
    处理指定文件夹内的所有图像和XML标注文件，并将增强后的图像保存到输出文件夹。
    
    :param folder_path: 包含图像和XML文件的文件夹路径
    :param output_folder: 增强后图像的保存路径
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            xml_path = os.path.join(folder_path, filename[:-4] + '.xml')
            
            # 读取图像和解析XML文件
            image = cv2.imread(img_path)
            img_height, img_width, _ = image.shape
            bboxes, _ = parse_xml(xml_path, (img_width, img_height))
            
            # 对每个边界框进行亮度增强
            for bbox in bboxes:
                image = enhance_brightness(image, bbox)
            
            # 定义输出路径
            output_path = os.path.join(output_folder, filename)
            # 保存增强后的图像
            cv2.imwrite(output_path, image)
            print(f'Saved enhanced image to {output_path}')

# 调用函数处理文件夹内的所有图像
folder_path = '.\input'  # 替换为你的图像和XML文件所在的文件夹路径
output_folder = '.\output'  # 替换为你希望保存增强图像的文件夹路径
if not os.path.exists(output_folder):
    os.makedirs(output_folder)  # 如果输出文件夹不存在，则创建它

process_images_and_annotations(folder_path, output_folder)