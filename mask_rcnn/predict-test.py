import os
import time
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision import transforms

from network_files import MaskRCNN
from backbone import resnet50_fpn_backbone
from draw_box_utils import draw_objs
from postprocessing import post_processing
from evaluation import evaluation

def create_model(num_classes, box_thresh=0.5):
    backbone = resnet50_fpn_backbone()
    model = MaskRCNN(backbone,
                     num_classes=num_classes,
                     rpn_score_thresh=box_thresh,
                     box_score_thresh=box_thresh)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main(name):
    num_classes = 1  # 不包含背景
    box_thresh = 0.8
    weights_path = "./save_weights/model_54.pth"
    img_path = "./new_test_result/img/"+name+".jpg"
    label_json_path = './pascal_voc_indices.json'
    xml_path = "./new_test_result/Annotations/"

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=num_classes + 1, box_thresh=box_thresh)

    # load train weights
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as json_file:
        category_index = json.load(json_file)

    # load image
    assert os.path.exists(img_path), f"{img_path} does not exits."
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # 获取图片名字
    img_name = os.path.basename(img_path)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_inference_end = time_synchronized()
        print("inference time: {}".format(t_inference_end - t_start))

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()
        predict_mask = predictions["masks"].to("cpu").numpy()
        predict_mask = np.squeeze(predict_mask, axis=1)  # [batch, 1, h, w] -> [batch, h, w]

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
            return

        plot_img = draw_objs(original_img,
                             boxes=predict_boxes,
                             classes=predict_classes,
                             scores=predict_scores,
                             category_index=category_index,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=14,
                             img_name=img_name)

        plt.imshow(plot_img)
        # plt.show()
        # 保存预测的图片结果
        plot_img.save("./new_test_result/ordinate_test/"+name+".jpg")

        process_boxes,postprocessing_time = post_processing(img_path, predict_boxes, name)
        # process_boxes = post_processing(img_path, process_boxes, name)  # 循环两遍，查缺补漏，如果需要的话
        
        # true_positives, true_num = evaluation(xml_path, process_boxes, name)    # 计算后处理后预测框的指标（二选一）
        true_positives, true_num = evaluation(xml_path, predict_boxes, name)  # 计算后处理前预测框的指标（二选一）

        # return true_num, len(process_boxes), true_positives, t_inference_end - t_start, postprocessing_time  # 计算后处理后预测框的指标（二选一）
        return true_num,len(predict_boxes),true_positives, t_inference_end - t_start, postprocessing_time  # 计算后处理前预测框的指标（二选一）


if __name__ == '__main__':
    # 打开文件
    file_path = "./new_test_result/else.txt"
    with open(file_path, 'r') as file:
        # 逐行读取文件内容，并去除每行末尾的换行符
        image_list = [line.strip() for line in file.readlines()]
    all_true_num = 0
    all_predict_num = 0
    true_predict_num = 0
    total_inference_time = 0
    total_postprocessing_time = 0

    for img_name in image_list:
        true_num, predict_num, true_predict, inference_time, postprocessing_time = main(img_name)
        print("真实目标数：", true_num)
        all_true_num += true_num
        all_predict_num += predict_num
        true_predict_num += true_predict
        total_inference_time += inference_time
        total_postprocessing_time += postprocessing_time

    # 计算查准率
    precision = true_predict_num / all_predict_num
    # 计算查全率
    recall = true_predict_num / all_true_num
    print("查准率：", precision)
    print("查全率：", recall)
    print("目标总数：", all_true_num, "预测总数：", all_predict_num, "预测正确数：", true_predict_num)
    print("总推理时间：", total_inference_time, "总后处理时间：", total_postprocessing_time)
    print("推理+后处理总时间：", total_inference_time + total_postprocessing_time)