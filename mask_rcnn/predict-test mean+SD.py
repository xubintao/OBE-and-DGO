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

def list_pth_files(directory):
    # 检查给定的路径是否存在
    if not os.path.exists(directory):
        print("The specified directory does not exist.")
        return []

    # 初始化一个空列表来存储.pth文件的路径
    pth_files = []

    # 遍历目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件后缀是否为.pth
            if file.endswith('.pth'):
                # 将.pth文件的完整路径添加到列表中
                pth_files.append(os.path.join(root, file))

    return pth_files

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


def main(name,weights_path):
    num_classes = 1  # 不包含背景
    box_thresh = 0.5
    # weights_path = "./save_weights-33(ostu-1.1）/model_392.pth"
    weights_path = weights_path
    img_path = "./Tunnel-C_test_result/img/"+name+".jpg"
    label_json_path = './pascal_voc_indices.json'
    xml_path = "./Tunnel-C_test_result/Annotations/"

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
        plot_img.save("./Tunnel-C_test_result/ordinate_test/"+name+".jpg")

        process_boxes1,postprocessing_time = post_processing(img_path, predict_boxes, name)
        process_boxes2,postprocessing_time = post_processing(img_path, process_boxes1, name)  # 循环两遍，查缺补漏

        # # 计算后处理后预测框的指标（二选一）
        # true_positives1, true_num = evaluation(xml_path, process_boxes1, name) 
        # true_positives2, true_num = evaluation(xml_path, process_boxes2, name) 
        # # 查准率
        # precision1=true_positives1/len(process_boxes1)
        # precision2=true_positives2/len(process_boxes2)
        # # 查全率
        # recall1=true_positives1/true_num
        # recall2=true_positives2/true_num
        # # f1_score
        # f1_score1=2*precision1*recall1/(precision1+recall1)
        # f1_score2=2*precision2*recall2/(precision2+recall2)
        # if f1_score1>f1_score2:
        #     print("后处理一次效果更好，返回结果")
        #     return true_num, len(process_boxes1), true_positives1, t_inference_end - t_start, postprocessing_time
        # else:
        #     print("后处理一次效果更好，返回结果")
        #     return true_num, len(process_boxes2), true_positives2, t_inference_end - t_start, postprocessing_time
        
        # 计算后处理前预测框的指标（二选一）
        true_positives, true_num = evaluation(xml_path, predict_boxes, name)  
        return true_num,len(predict_boxes),true_positives, t_inference_end - t_start, postprocessing_time     


if __name__ == '__main__':
    # 打开文件
    file_path = "./Tunnel-C_test_result/else.txt"
    with open(file_path, 'r') as file:
        # 逐行读取文件内容，并去除每行末尾的换行符
        image_list = [line.strip() for line in file.readlines()]
    
    # 将'your_directory_path'替换为你的目标文件夹路径
    your_directory_path = './pth_files_directory'
    pth_files_list = list_pth_files(your_directory_path)
    
    list_all_true_num = []
    list_all_predict_num = []
    list_true_predict_num = []
    list_total_inference_time = []
    list_total_postprocessing_time = []

    for repeat in range(len(pth_files_list)):
        print(pth_files_list[repeat],"--------------------------------")
        all_true_num = 0
        all_predict_num = 0
        true_predict_num = 0
        total_inference_time = 0
        total_postprocessing_time = 0

        for img_name in image_list:
            true_num, predict_num, true_predict, inference_time, postprocessing_time = main(img_name,pth_files_list[repeat])
            print("真实目标数：", true_num)
            all_true_num += true_num
            all_predict_num += predict_num
            true_predict_num += true_predict
            total_inference_time += inference_time
            total_postprocessing_time += postprocessing_time
        
        list_all_true_num.append(all_true_num)
        list_all_predict_num.append(all_predict_num)
        list_true_predict_num.append(true_predict_num)
        list_total_inference_time.append(total_inference_time)
        list_total_postprocessing_time.append(total_postprocessing_time)

    print(list_all_true_num)
    print(list_all_predict_num)
    print(list_true_predict_num)
    print(list_total_inference_time)
    print(list_total_postprocessing_time)

    # 将列表转换为NumPy数组,方便相除
    list_all_true_num=np.array(list_all_true_num)
    list_all_predict_num=np.array(list_all_predict_num)
    list_true_predict_num=np.array(list_true_predict_num)
    list_total_inference_time=np.array(list_total_inference_time)
    list_total_postprocessing_time=np.array(list_total_postprocessing_time)

    # 计算查准率
    list_precision = list_true_predict_num / list_all_predict_num
    # 计算查全率
    list_recall = list_true_predict_num / list_all_true_num
    # 计算f1_score
    list_f1_score = 2 * list_precision * list_recall / (list_precision + list_recall)
    # 计算推理+后处理总时间
    list_total_time = list_total_inference_time + list_total_postprocessing_time

    # 计算均值
    mean_list_precision = np.mean(list_precision)
    mean_list_recall = np.mean(list_recall)
    mean_list_f1_score = np.mean(list_f1_score)
    mean_inference_time = np.mean(list_total_inference_time)
    mean_postprocessing_time = np.mean(list_total_postprocessing_time)
    mean_total_time = np.mean(list_total_time)

    # 计算标准差
    std_list_precision = np.std(list_precision)
    std_list_recall = np.std(list_recall)
    std_list_f1_score = np.std(list_f1_score)
    std_inference_time = np.std(list_total_inference_time)
    std_postprocessing_time = np.std(list_total_postprocessing_time)
    std_total_time = np.std(list_total_time)

    print(pth_files_list)  # 打印.pth文件列表
    print("查准率均值：", mean_list_precision,"查准率标准差：", std_list_precision)
    print("查全率均值：", mean_list_recall,"查全率标准差：", std_list_recall)
    print("f1均值：", mean_list_f1_score,"f1标准差：", std_list_f1_score)
    print("推理时间均值：", mean_inference_time, "推理时间标准差：", std_inference_time)
    print("后处理时间均值：", mean_postprocessing_time, "后处理时间标准差：", std_postprocessing_time)
    print("推理+后处理时间均值：", mean_total_time, "推理+后处理时间标准差：", std_total_time)
