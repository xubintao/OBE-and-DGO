import xmltodict  
import numpy as np  
  
def parse_voc_xml(xml_file):  
    with open(xml_file, 'r',encoding="utf-8") as f:  
        xml_dict = xmltodict.parse(f.read())  
      
    boxes = []  
    for obj in xml_dict['annotation']['object']:  
        xmin = int(obj['bndbox']['xmin'])  
        ymin = int(obj['bndbox']['ymin'])  
        xmax = int(obj['bndbox']['xmax'])  
        ymax = int(obj['bndbox']['ymax'])  
        boxes.append([xmin, ymin, xmax, ymax])  
      
    return boxes  
  
def calculate_iou(boxA, boxB):  
    # 确定交集区域的 (x, y, w, h)  
    xA = max(boxA[0], boxB[0])  
    yA = max(boxA[1], boxB[1])  
    xB = min(boxA[2], boxB[2])  
    yB = min(boxA[3], boxB[3])  
      
    # 计算交集面积  
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)  
      
    # 计算两个框的面积  
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)  
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)  
      
    # 计算交并比  
    iou = interArea / float(boxAArea + boxBArea - interArea)  
      
    # 返回iou值  
    return iou  
  
def calculate_precision(gt_boxes, pred_boxes, iou_threshold=0.4):  
    # 初始化TP（真正例）和FP（假正例）计数  
    true_positives = 0  
    false_positives = len(pred_boxes)  
      
    # 对预测方框进行遍历  
    for pred_box in pred_boxes:  
        max_iou = 0  
        # 计算预测方框与所有真实方框的IoU  
        for gt_box in gt_boxes:  
            iou = calculate_iou(pred_box, gt_box)  
            if iou > max_iou:  
                max_iou = iou  
          
        # 如果最大IoU超过阈值，则认为是真正例  
        if max_iou >= iou_threshold:  
            true_positives += 1  
            false_positives -= 1  # 修正FP计数，因为该预测方框现在是TP  
      
    # 避免除以零的情况  
    if true_positives + false_positives == 0:  
        return 0  
      
    # 计算查准率  
    precision = true_positives / (true_positives + false_positives)  
    return precision,true_positives  

def evaluation(xml_path,pred_boxes,name):
    """
        pred_boxes = [[10, 10, 50, 50], [60, 60, 100, 100]]  # 示例预测方框，格式为[xmin, ymin, xmax, ymax]  
    """
    xml_path=xml_path+name+".xml"
    gt_boxes = parse_voc_xml(xml_path)  # 替换为真实的标注XML文件路径  
    precision,true_positives = calculate_precision(gt_boxes, pred_boxes)  
    print(f'{name}_Precision: {precision:.2f}')
    return true_positives,len(gt_boxes)