import json
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from pyod.models.knn import KNN
import pwlf
import matplotlib.patches as patches
from scipy.optimize import curve_fit
import torch
from torchvision import transforms
import PIL.ImageDraw as ImageDraw

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
def read_boxes_from_file(file_path):
    """
    从给定文件路径读取方框坐标列表
    """
    boxes = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                box_line = json.loads(line.strip())
                boxes.append(box_line)
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return []
    return boxes

def open_image(file_path):
    """
    打开给定文件路径的图像，并返回图像对象和尺寸
    """
    try:
        img = Image.open(file_path)
        width, height = img.size
        return img, width, height
    except Exception as e:
        print(f"打开图像时发生错误: {e}")
        return None, None, None

def process_boxes(boxes, width, height):
    """
    处理方框列表，按x坐标排序、分组
    """
    center_xs = [(box[0] + box[2]) / 2 for box in boxes]
    sorted_boxes = [box for _, box in sorted(zip(center_xs, boxes))]

    top_boxes = []
    bottom_boxes = []
    for box in sorted_boxes:
        if box[1] <= height - box[3]:
            if 160<(box[1]+box[3])/2<220:
                top_boxes.append(box)
        else:
            if 410<(box[1]+box[3])/2<470:
                bottom_boxes.append(box)
        # 若为Tunnel-B和C。图片则改为该区间
        # if box[1] <= height - box[3]:
        #     if 70<(box[1]+box[3])/2<130:
        #         top_boxes.append(box)
        # else:
        #     if 240<(box[1]+box[3])/2<300:
        #         bottom_boxes.append(box)
    
    return (top_boxes,bottom_boxes)

def calculate_max_width(boxes):
    """
    计算方框列表的最大方框宽度
    """
    widths = [box[2] - box[0] for box in boxes]
    return np.max(widths)

def perform_dbscan_clustering(data, epsilon, min_samples=1):
    """
    使用DBSCAN算法对一维数据进行聚类，并返回聚类标签。

    参数:
    - data: 一维数据列表或数组
    - epsilon: DBSCAN算法中的eps参数
    - min_samples (可选): DBSCAN算法中的min_samples参数，默认为1

    返回:
    - labels: 聚类后的标签数组
    """
    data_array = np.array(data)
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
    dbscan.fit(data_array.reshape(-1, 1))
    labels_local = dbscan.labels_
    return labels_local

def find_stable_boxes(labels):
    """
    找到稳定方框的索引（一个簇只有这一个方框）。
    :param labels: 方框的标签列表。
    :return: 稳定方框的索引列表。
    """
    stable_boxes = []  # 方框的索引
    for i in range(1, len(labels) - 1):
        if labels[i] != labels[i - 1] and labels[i] != labels[i + 1]:
            stable_boxes.append(i)
    # 添加边界条件索引（如果适用）
    if labels[0] != labels[1]:
        stable_boxes.insert(0, 0)
    if labels[-2] != labels[-1]:
        stable_boxes.append(len(labels) - 1)
    return stable_boxes

def remove_intersecting_boxes(boxes):
    """
    从稳定方框列表中移除与后续方框在x轴上相交的方框。
    :param boxes: 方框的坐标列表。
    :param stable_boxes: 稳定方框的索引列表。
    :return: 无相交的稳定方框的索引列表。
    """
    non_intersecting_boxes = set(range(len(boxes)))
    for i in range(len(boxes)-1):  # 避免数组越界，不包括最后一个元素
        overlap=boxes[i][2] - boxes[i + 1][0]
        min_size=min(boxes[i][2]-boxes[i][0],boxes[i+1][2]-boxes[i+1][0])
        # 若重叠度大于百分之十则删除
        if overlap>0 and overlap/min_size>0.1:
            non_intersecting_boxes.discard(i)
            non_intersecting_boxes.discard(i + 1)
    return list(non_intersecting_boxes)

def calculate_spacing_with_centers(sorted_boxes,fixed_boxes):
    # 计算相邻方框在x轴方向上的间距及其前一个方框的中心坐标
    spacings = []
    centers = []
    for i in range(1,len(fixed_boxes)):
        forward = fixed_boxes[i-1]
        backward = fixed_boxes[i]
        spacing = (sorted_boxes[backward][0] + sorted_boxes[backward][2]) / 2  - (sorted_boxes[forward][0] + sorted_boxes[forward][2]) / 2   # 假设间距是右方框到左方框的中心坐标的差值
        center = (sorted_boxes[forward][0] + sorted_boxes[forward][2]) / 2  # 计算前一个方框的中心坐标
        spacings.append(spacing)
        centers.append(center)

    # 获得末尾方框的坐标
    last_num=fixed_boxes[-1]
    centers.append((sorted_boxes[last_num][0] + sorted_boxes[last_num][2]) / 2)

    return spacings, centers


def remove_global_outliers(data, threshold=2.5):
    # 使用KNN算法去除全局离群点并记录索引
    knn = KNN(n_neighbors=min(5,len(data)-1))  # 假设KNN类已定义
    knn.fit(data.reshape(-1, 1))
    scores = knn.decision_function(data.reshape(-1, 1))

    # # 计算数据集的上四分位数
    # upper_quartile = np.percentile(data, 75)
    median_value=np.mean(data)*1.9
    # print(median_value)
    # 找到离群点并且其值大于上四分位数的索引
    global_outlier_indices = np.where((scores > threshold) & (data > median_value))[0]

    # 删除这些离群点
    filtered_data = np.delete(data, global_outlier_indices, axis=0)

    return filtered_data, global_outlier_indices

# def remove_global_outliers(data, iqr_threshold=1.5):
#     # 计算数据集的一阶四分位数（Q1）和三阶四分位数（Q3）
#     q1 = np.percentile(data, 25)
#     q3 = np.percentile(data, 75)

#     # 计算IQR
#     iqr = q3 - q1

#     # 定义上下界
#     lower_bound = q1 - iqr_threshold * iqr
#     upper_bound = q3 + iqr_threshold * iqr

#     # 找到超出上下界的离群点索引
#     outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]

#     # 删除离群点
#     filtered_data = np.delete(data, outlier_indices, axis=0)

#     return filtered_data, outlier_indices

def calculate_error_sum(data1, data2):
    # 定义线性函数模型
    def linear_func(x, a, b):
        return a * x + b

    # 对第一组数据进行线性拟合
    popt1, _ = curve_fit(linear_func, np.arange(len(data1)), data1)

    # 对第二组数据进行线性拟合
    popt2, _ = curve_fit(linear_func, np.arange(len(data2)), data2)

    # 计算误差平方和的和
    error_sum = np.sum((data1 - linear_func(np.arange(len(data1)), *popt1)**2)) + np.sum((data2 - linear_func(np.arange(len(data2)), *popt2)**2))

    return error_sum

def detect_change_points(new_centers,y_data):
    """
        使用pwlf包检测局部分段线性变化
        选用aic或bic对段数不同时的数据分段进行打分，选择打分低的分段数，并依照分段数选择分段点
        :param x_data: x轴坐标列表。
        :param y_data: y轴坐标列表。
    """
    # 定义计算AIC和BIC的辅助函数
    def compute_criteria(n_segments, residuals, df_model, log_likelihood):
        # df_model是模型的自由度，等于分段数加1（每个分段有一个斜率和一个截距，共两个参数，最后一段只一个截距）
        df_model += 1  # 因为我们还加上了全局截距项
        rss = np.sum(residuals**2)  # 残差平方和

        # 计算AIC和BIC
        aic = 2 * df_model - 2 * log_likelihood + 2 * df_model * n_segments / len(y_data)
        bic = np.log(len(y_data)) * df_model - 2 * log_likelihood + 2 * df_model * n_segments / len(y_data)

        return aic, bic

    x_data=list(range(len(new_centers[:-1])))
    # 初始化pwlf对象
    my_pwlf = pwlf.PiecewiseLinFit(x_data, y_data)
    true_pwlf=pwlf.PiecewiseLinFit(new_centers[:-1], y_data)
    # 设定最大分段数
    max_segments = 2  # 根据实际情况调整

    aic_values = []
    bic_values = []
    if len(x_data)<=2:
        best_n_segments=1
    else:
        best_n_segments = None
    best_aic = float('inf')
    best_bic = float('inf')

    # 循环尝试不同分段数
    for n_segments in range(1, min(len(x_data)-1, max_segments+1)):
        # 拟合模型
        my_pwlf.fit(n_segments)

        # 计算每个数据点的预测值
        y_pred = my_pwlf.predict(x_data)

        # 计算每个数据点的残差
        residuals = y_data - y_pred

        # 计算似然函数值
        log_likelihood = -len(y_data)/2 * np.log(2*np.pi*residuals.var()) - len(y_data)/2

        # 计算AIC和BIC
        aic, bic = compute_criteria(n_segments, residuals, n_segments, log_likelihood)

        # 存储AIC和BIC值
        aic_values.append(aic)
        bic_values.append(bic)
        
        # 更新最优分段数及其对应的AIC和BIC
        # if aic < best_aic:
        #     best_aic = aic
        #     best_n_segments = n_segments
        if bic < best_bic:
            best_bic = bic
            best_n_segments = n_segments

    # 输出最优分段数及其对应的AIC和BIC
    # print(f"最优分段数(AIC): {best_n_segments}，AIC值: {best_aic}")
    # print(f"最优分段数(BIC): {best_n_segments}，BIC值: {best_bic}")

    # 使用最优分段数重新拟合模型
    breakpoints=my_pwlf.fit(best_n_segments)
    true_breakpoints=true_pwlf.fit(best_n_segments)
    # print("ture_breakpoints:",true_breakpoints)
    # print(new_centers)
    # print("x_data:",x_data)
    # print("breakpoints:",breakpoints)

    breakpoint_indices = np.searchsorted(x_data, breakpoints,side="right")

    # 去除重复的分段点索引并保持顺序
    unique_breakpoint_indices = [idx for i, idx in enumerate(breakpoint_indices) if idx not in breakpoint_indices[:i]]

    # # 处理边界情况
    # if unique_breakpoint_indices[-1] != len(x_data):
    #     unique_breakpoint_indices[-1] = unique_breakpoint_indices[-1] + 1
    unique_breakpoint_indices[0] = 0
    # print("unique_breakpoint_indices:",unique_breakpoint_indices)

    new_breakpoints=unique_breakpoint_indices
    result0 = float('inf')
    if len(breakpoints)==3:
        for i in range(unique_breakpoint_indices[1],unique_breakpoint_indices[-1]-1):
            list1=y_data[0:i]
            list2=y_data[i:]
            result = calculate_error_sum(list1, list2)
            if result < result0:
                result0 = result
                new_breakpoints[1]=i
    
    # print("new_breakpoints:",new_breakpoints)

    # 每一点的索引属于右边那一段的值
    return new_breakpoints

def remove_local_outliers(spacings, breakpoints,original_spacings,threshold=2):
    """
        在分段基础上进一步剔除局部离群点，并记录索引
    """
    local_outlier_indices = []
    segmented_spacings = []
    # print(breakpoints)
    for i in range(1,len(breakpoints)):
        start = breakpoints[i - 1] 
        end = breakpoints[i]
        segment = spacings[start:end]
        # 每一段需要至少三个数据点才有剔除离群值的意义
        # 三个值代表索引差了2
        if end-start <= 2:
            filtered_segment = [x for idx, x in enumerate(segment)]
        else:
            q1, q3 = np.percentile(segment, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr

            # 需要保留的spacing
            # 需要删除的索引
            # 只去除因大值而偏离的离群点
            filtered_segment = [x for idx, x in enumerate(segment) if  x <= upper_bound]
            outlier_segment_values = [val for idx, val in enumerate(segment) if val > upper_bound]
            for idx, val in enumerate(original_spacings):
                for i in range(len(outlier_segment_values)):
                    if outlier_segment_values[i] == val:
                        local_outlier_indices.append(idx)


        segmented_spacings.extend(filtered_segment)

    return np.array(segmented_spacings), np.array(local_outlier_indices)

def filter_outliers(spacings, centers, fixed_boxes):
    '''
        获得滤除全局和局部离群值的spacings和对应头部方框的坐标和索引
    '''
    spacings=np.array(spacings)
    # print("spacings:",spacings)
    # 去除全局离群值并获取全局离群值在spacings的索引
    global_out_spacings, global_outlier_indices = remove_global_outliers(spacings)
    
    # 排除了指定索引对应的全局离群值，获得剩下的中心坐标
    new_centers= [value for i, value in enumerate(centers) if i not in global_outlier_indices]
    # print(global_out_spacings)
    # 检测分段线性回归之间的趋势变点
    breakpoints= detect_change_points(new_centers,global_out_spacings)
    # print("breakpoints:",breakpoints)
    # 获得中间断点在fixed_boxes中的索引
    fixed_breakpoints=breakpoints[:]
    if len(fixed_breakpoints)==3:
        for i in range(1,len(fixed_breakpoints)):
            count=len([x for x in global_outlier_indices if x < fixed_breakpoints[i]])
            if count>0:
                fixed_breakpoints[i]=fixed_breakpoints[i]+count
    # 去除局部离群值并获取局部离群值在spacings的索引
    filtered_spacings, local_outlier_indices = remove_local_outliers(global_out_spacings, breakpoints,spacings)

    # 去除局部离群值之后重新调整breakpoints
    # 有多少个值小于断点
    for i in range(len(breakpoints)):
        count = len([x for x in local_outlier_indices if x < breakpoints[i]])
        breakpoints[i] = breakpoints[i] - count
    # 输出结果
    # print("filtered spacings:", filtered_spacings)
    # print("Global outlier indices (in original spacings):", global_outlier_indices)
    # print("Local outlier indices (in original spacings):", [idx for idx in local_outlier_indices if idx not in global_outlier_indices])

    # 注意：local_outlier_indices 需要排除已经在全局离群点处理阶段被移除的索引
    # 先检查 local_outlier_indices 是否为空
    if local_outlier_indices.size > 0:
        # 进行合并操作，获得所有离群点在spacings的索引
        index_to_remove = np.concatenate((local_outlier_indices, global_outlier_indices))
    else:
        index_to_remove = global_outlier_indices
    
    # 获得过滤后的中心坐标，即正常间距头部方框中心坐标
    filtered_centers = [item for i, item in enumerate(centers) if i not in index_to_remove][:-1]
    # 获得过滤后的方框索引，即正常间距头部方框在sorted_boxs中的索引
    filtered_boxes = [item for i, item in enumerate(fixed_boxes) if i not in index_to_remove][:-1]

    return filtered_spacings, filtered_centers, filtered_boxes,breakpoints,index_to_remove,fixed_breakpoints
    
def visualize_boxes_and_spacings(img, boxes,fixed_boxes, filtered_centers, filtered_spacings):
    """
    可视化稳定方框在原始图像中的位置以及间距随中心位置的变化。
    
    :param img: PIL Image 对象，原始图像
    :param boxes: list of lists，每个内部列表包含方框坐标信息 [left, top, right, bottom]
    :param fixed_boxes: list，经过处理的稳定方框的索引列表
    :param filtered_centers: list，稳定方框的中心x坐标
    :param filtered_spacings: list，与中心位置对应的过滤后的间距数据
    """

    # 将PIL图像转换为numpy数组以便在matplotlib中显示
    img_array = np.array(img)

    # 创建一个新的figure，并设置子图布局
    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # 第一个子图 - 显示原始图像及稳定方框
    axs[0].imshow(img_array)
    axs[0].set_title('Original Image with Stable Boxes')
    for index in fixed_boxes:
        axs[0].add_patch(plt.Rectangle((boxes[index][0], boxes[index][3]), 
                                       boxes[index][2] - boxes[index][0], 
                                       boxes[index][1] - boxes[index][3],
                                       fill=False, edgecolor='red', linewidth=2))

    # 第二个子图 - 显示间距随中心位置的变化
    print("filtered_centers:",filtered_centers)
    print("filtered_spacings:",filtered_spacings)
    axs[1].scatter(filtered_centers, filtered_spacings, label='Spacing data')
    axs[1].plot(filtered_centers, filtered_spacings, '-', label='Spacing trend')
    axs[1].set_xlabel('Box Center Position')
    axs[1].set_ylabel('Spacing')
    axs[1].legend()
    axs[1].set_title('Spacing vs. Box Center Position')

    plt.tight_layout()  # 紧凑布局以避免子图之间的重叠
    plt.show()

def find_middle_values(sorted_list, target_value):
    """
    如果目标值位于列表中两个相邻元素之间，则返回这两个元素
    """
    for i in range(len(sorted_list) - 1):
        if sorted_list[i] <= target_value <= sorted_list[i + 1]:
            return sorted_list[i], sorted_list[i + 1]
    return None


# def get_neighbouring_elements(filtered_spacings,index,breakpoints,node):
#     """
#     获取给定索引附近的元素列表。
#     :param filtered_spacings: list，过滤后的间距数据
#     :param index: int，要获取附近的元素列表的索引
#     :param breakpoints: list，分段线性回归的断点列表
#     :return: list，给定索引附近的元素列表,左右各三个元素
#     """
#     if node:
#         return filtered_spacings[index-3:index+3]
#     else:
#         #计算索引的上下边界
#         left,right=find_middle_values(breakpoints, index)
#         print("left,right:",left,right)
#         lst=filtered_spacings[left:right]
#         i=index-left
#         if 0 <= i < len(lst) - 2:
#             return lst[max(0, i-3):min(len(lst), i+3+1)]
#         else:
#             # 处理边界情况，确保不会越界
#             if i < 0:
#                 return lst[0:i+6]  # 当i为负数时，返回列表开头到不超过i+6位置的所有元素
#             elif i >= len(lst) - 2:
#                 return lst[i-3:]   # 当i接近列表结尾时，返回从i-3到列表结尾的所有元素
def get_neighbouring_elements(filtered_spacings, index, breakpoints, node):
    """
    获取给定索引附近的元素列表，总计至少包含六个元素，优先从中心点两侧获取，不足部分从另一边补充。
    
    :param filtered_spacings: list，过滤后的间距数据
    :param index: int，要获取附近的元素列表的索引
    :param breakpoints: list，分段线性回归的断点列表
    :return: list，给定索引附近的元素列表，总计至少包含六个元素
    """
    if node:
        left=max(0,index-3)
        right=min(len(filtered_spacings)-1,index+3)
        return filtered_spacings[left:right]

    else:
        # 计算索引的上下边界
        left, right = find_middle_values(breakpoints, index)
        lst = filtered_spacings[left:right]
        i = index - left

        # 确保选取范围包含至少六个元素
        half_window_size = 3
        start, end = max(0, i - half_window_size), min(len(lst), i + half_window_size)
        window_size = end - start
        
        # 如果窗口大小小于六，则尝试扩展窗口以包含更多元素
        while window_size < 6:
            if start == 0 and end == len(lst):  # 已经达到边界条件
                break
            elif start == 0:  # 左边已无元素可补充
                end += 1
            elif end == len(lst):  # 右边已无元素可补充
                start -= 1
            else:  # 两边都有元素可以补充
                if abs(start - i) < abs(end - i):  # 更靠近左边，补充右边
                    end += 1
                else:  # 更靠近右边，补充左边
                    start -= 1
            window_size = end - start
            
        return lst[start:end]

def find_duplicate_indices(lst):
    duplicates_index = {}
    for index, item in enumerate(lst):
        if item not in duplicates_index:
            duplicates_index[item] = [index]
        else:
            duplicates_index[item].append(index)
    # 返回只有重复元素及其索引的字典
    return {k: v for k, v in duplicates_index.items() if len(v) > 1}


# 获取列表中离给定值a最近的值
def find_closest_value(lst, a):
    closest = None
    min_distance = float('inf')

    for num in lst:
        distance = abs(num - a)
        if distance < min_distance:
            closest = num
            min_distance = distance

    return closest

def new_section_centers_generator(full_centers,new_centers,epsilon):
        old_centers=full_centers[len(new_centers):]
        new_section_centers=[]

        for new_center in new_centers:
            # 找到离平均框最近的原始框
            closest=find_closest_value(old_centers, new_center)
            # 如果有原始框
            if closest:
                # 若最近框在1/2最大框之内，则被选中，否则选择平均框
                if abs(closest-new_center) < epsilon:
                    new_section_centers.append(closest)
                else:
                    new_section_centers.append(new_center)
            else:
                new_section_centers.append(new_center)
        return new_section_centers

def section_centers_generator(centers,generate_num,i,epsilon,old_centers):
    avg_spacing=(centers[i]-centers[i-1])/(generate_num+1)
    new_centers=[]
    # 短距离内利用平均间距生成新方框
    for j in range(1,generate_num+1):
        new_centers.append(centers[i-1]+j*avg_spacing)
    # 或者应用线性趋势
        
    # 和原来因为聚类、重叠删除的簇进行聚类中心，选出最优中心
    full_centers=new_centers+old_centers
    new_section_centers=new_section_centers_generator(full_centers,new_centers,epsilon)
    
    return new_section_centers

def boxes_generator(index_to_remove,fixed_boxes,centers,filtered_boxes,filtered_spacings,labels,breakpoints,epsilon,centers_x,sorted_boxes,img_width,filtered_centers,fixed_breakpoints):
    """
    生成最终的稳定方框列表。
    :param filtered_boxes: list，经过过滤的间距的前端方框的索引列表
    :param filtered_centers: list，经过过滤的间距的前端方框的中心x坐标
    :param filtered_spacings: list，经过过滤的间距数据
    :return: list，生成的新方框列表
    """
    # 生成需要添加方框的间距
    generate_spacings=[]

    # 新的方框列表
    new_sorted_boxes=[]

    # 左边的边界
    # left_bound=286/4
    left_bound=0
    # 右边的边界
    # right_bound=img_width-145/4
    right_bound=img_width

    # 无断点时，设断点在fixed_boxes中的索引为0
    bkp_fixed_index=0

    # 遍历所有异常索引，i是fixed_boxes/centers的索引
    for i in index_to_remove:
        generate_num=0
        #计算i+1到i的间距
        generate_spacings.append(centers[i+1]-centers[i])
        # fixed_boxes和filtered_boxes的值都是sorted_boxes的索引，filtered_boxes索引则可以用来搜索filtered_spacings
        # 寻找fixed_boxes[i]在sorted_boxes中离的最近的filtered_boxes的值，获取filtered_boxes对应的索引
        # 为什么不直接用i来寻找list(range(filtered_boxes))中最近的值,因为filtered_boxes去掉了长间距
        index=np.searchsorted(filtered_boxes,fixed_boxes[i],side="right")

        node=False
        if len(breakpoints)==3:
            if i==fixed_breakpoints[1]:
                node=True

        # breakpoints:分段的端点的在filtered_spacings/filtered_centers/filtered_boxes中的索引列表
        # 寻找分段后周围(左右各三间距)的最小值和最大值,附近6个元素
        neighbours = get_neighbouring_elements(filtered_spacings,index,breakpoints,node)
        # print("neighbours:",neighbours)
        max_spacing=max(neighbours)
        min_spacing=min(neighbours)
        #长间距/周围间距的最小值和最大值，获得方框数量范围
        max_num=generate_spacings[-1]/min_spacing-1
        min_num=generate_spacings[-1]/max_spacing-1
        if min_num-int(min_num)<0.1:
            min_num=int(min_num)
        # 获取范围内的整数，判断列表是否非空
        integers_between=list(range(int(min_num-0.1)+1,int(max_num)+1))
        #范围内有整数
        # print("labels:",labels)
        # print("簇数：",labels[fixed_boxes[i+1]]-labels[fixed_boxes[i]]-1)
        # print("整数范围min-max：",min_num,max_num)
        if integers_between:
            # 若簇数在范围内，取簇数
            if min_num <=labels[fixed_boxes[i+1]]-labels[fixed_boxes[i]]-1<=max_num:
                generate_num=labels[fixed_boxes[i+1]]-labels[fixed_boxes[i]]-1
            # 若范围外，取范围内大小接近簇数的整数
            elif min_num > labels[fixed_boxes[i+1]]-labels[fixed_boxes[i]]-1:
                generate_num=integers_between[0]
            else:
                generate_num=integers_between[-1]
            # 若范围内无整数，取离范围最近的整数，即将最大值与大整数的间距和最小值与小整数的间距进行比较
        else:
            if abs(max_num-(int(max_num)+1)) < abs(min_num-int(min_num)):
                generate_num=int(max_num)+1
            else:
                generate_num=int(min_num)

        # print("generate_num:",generate_num)

        # 范围内对应的簇列表
        label_list=labels[fixed_boxes[i-1]+1:fixed_boxes[i]]
        # print(label_list)
        # 范围内对应的中心列表
        old_centers=centers_x[fixed_boxes[i]+1:fixed_boxes[i+1]]

        # 生成具体方框的x列表
        new_section_centers=section_centers_generator(centers,generate_num,i+1,epsilon,old_centers)

        # 平均y值、宽度和高度
        data_for_boxes=sorted_boxes[fixed_boxes[i]:fixed_boxes[i+1]]
        center_y = np.mean([(box[1] + box[3]) / 2 for box in data_for_boxes])
        height=np.median([(box[3]-box[1]) for box in data_for_boxes])
        width=np.median([(box[2]-box[0]) for box in data_for_boxes])

        # 生成具体方框数据
        new_section_boxes=[]
        for center_x in new_section_centers:
            left,top,right,bottom=center_x-width/2,center_y-height/2,center_x+width/2,center_y+height/2
            new_section_boxes.append([left,top,right,bottom])

        # 添加进boxes里
        for box in new_section_boxes:
            new_sorted_boxes.append(box)

    # 获取除开头和结尾的boxes的间距
    # 只获取信任boxes
    del_sorted_boxes0=[sorted_boxes[i] for i in fixed_boxes]
    # 重新排序
    new_sorted_boxes0=new_sorted_boxes+del_sorted_boxes0
    center_xs0 = sorted([(box[0] + box[2]) / 2 for box in new_sorted_boxes0])
    middle_spacings = [center_xs0[i] - center_xs0[i-1] for i in range(1, len(center_xs0))]


    # 初始化    
    bkp_start=len(middle_spacings)
    bkp_end=0
    # 生成断点前后的点在middle_spacings中的索引
    if bkp_fixed_index !=0:
        bkp_fixed_value=fixed_boxes[bkp_fixed_index]
        bkp_fixed_value_start=fixed_boxes[bkp_fixed_index-1]
        bkp_start_box=sorted_boxes[bkp_fixed_value_start] 
        bkp_end_box=sorted_boxes[bkp_fixed_value]
        bkp_start_center=(bkp_start_box[0]+bkp_start_box[2])/2
        bkp_end_center=(bkp_end_box[0]+bkp_end_box[2])/2
        bkp_start=center_xs0.index(bkp_start_center)-1
        bkp_end=center_xs0.index(bkp_end_center)+1

    #在第一个方框前和最后一个方框后生成具体方框数据并加入列表
    # 直接以ps取真正的边界，图片则要求必须大小固定
    # 判断条件没什么含义，就是默认生成方框
    if fixed_boxes[0] !=-1:
        # 生成平均间距
        start_neighbours = middle_spacings[0:min(3,bkp_start)]
        start_avg_spacing=np.mean(start_neighbours)
        # 利用平均间距生成新方框
        new_start_centers=[]
        j=0
        while True:
            j=j+1
            if centers[0]-j*start_avg_spacing < left_bound:
                break
            new_start_centers.append(centers[0]-j*start_avg_spacing)
        # 因为没有知道开头的generate_num,只能让平均方框数量为generate_num
        # 和原来因为聚类、重叠删除的簇进行聚类中心，选出最优中心
        # 范围内对应的中心列表
        if j >1:
            old_start_centers=centers_x[0:fixed_boxes[0]]
            full_start_centers=new_start_centers+ old_start_centers
            new_section_centers=new_section_centers_generator(full_start_centers,new_start_centers,epsilon)
            # 平均y值、宽度和高度
            data_for_boxes=new_sorted_boxes0[0:min(3,bkp_start)]
            center_y = np.mean([(box[1] + box[3]) / 2 for box in data_for_boxes])
            height=np.median([(box[3]-box[1]) for box in data_for_boxes])
            width=np.median([(box[2]-box[0]) for box in data_for_boxes])
            # 生成具体方框数据
            new_section_boxes=[]
            for center_x in new_section_centers:
                left,top,right,bottom=center_x-width/2,center_y-height/2,center_x+width/2,center_y+height/2
                new_section_boxes.append([left,top,right,bottom])

            # 添加进boxes里
            for box in new_section_boxes:
                new_sorted_boxes.append(box)

    # 判断条件没什么含义，就是默认生成方框
    if fixed_boxes[-1]  != 0:
        #说明fixed_boxes后面还有空白区域，需要添加一个或多个框
        # 生成平均间距
        end_neighbours = middle_spacings[max(bkp_end-len(middle_spacings),-3):]
        end_avg_spacing=np.mean(end_neighbours)

        # 利用平均间距生成新方框
        new_end_centers=[]
        j=0
        while True:
            j=j+1
            if centers[-1]+j*end_avg_spacing > right_bound:
                break
            new_end_centers.append(centers[-1]+j*end_avg_spacing)
        if j > 1:
            # 因为没有知道开头的generate_num,只能让平均方框数量为generate_num
            # 和原来因为聚类、重叠删除的簇进行聚类中心，选出最优中心
            # 范围内对应的中心列表
            old_end_centers=centers_x[fixed_boxes[-1]+1:]
            full_end_centers=new_end_centers+ old_end_centers
            new_section_centers=new_section_centers_generator(full_end_centers,new_end_centers,epsilon)
            # 平均y值、宽度和高度
            data_for_boxes=new_sorted_boxes0[max(bkp_end-len(middle_spacings),-3):]
            center_y = np.mean([(box[1] + box[3]) / 2 for box in data_for_boxes])
            height=np.median([(box[3]-box[1]) for box in data_for_boxes])
            width=np.median([(box[2]-box[0]) for box in data_for_boxes])
            # 生成具体方框数据
            new_section_boxes=[]
            for center_x in new_section_centers:
                left,top,right,bottom=center_x-width/2,center_y-height/2,center_x+width/2,center_y+height/2
                new_section_boxes.append([left,top,right,bottom])

            # 添加进boxes里
            for box in new_section_boxes:
                new_sorted_boxes.append(box)

    # 只获取信任boxes
    del_sorted_boxes=[sorted_boxes[i] for i in fixed_boxes]
    # 重新排序
    new_sorted_boxes=new_sorted_boxes+del_sorted_boxes
    center_xs = [(box[0] + box[2]) / 2 for box in new_sorted_boxes]
    new_boxes = [box for _, box in sorted(zip(center_xs, new_sorted_boxes))]
    # 让方框限制在图片边界内
    if new_boxes[0][0] < left_bound:
        new_boxes[0][0]=left_bound
    if new_boxes[-1][2] > right_bound:
        new_boxes[-1][2]=right_bound

    # print("新的方框数:",len(new_boxes))
    return new_boxes

def visualize_boxes(img,boxes, output_path=None,show_after_save=None):
    """
    可视化方框在原始图像中的位置
    :param img: PIL Image 对象，原始图像
    :param boxes: list of lists，每个内部列表包含方框坐标信息 [left, top, right, bottom]
    :param output_path: str, 输出图像保存路径，若为空则仅显示图像
    """

    # dpi_factor = 1  # 根据需求调整
    # img_array = np.array(img)

    # fig, ax = plt.subplots(figsize=(img.size[0]/dpi_factor, img.size[1]/dpi_factor))
    # ax.imshow(img_array)
    # ax.axis('off')
    # ax.set_position([0, 0, 1, 1])

    # for box in boxes:
    #     left, top, right, bottom = box[0], box[1], box[2], box[3]
    #     rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='green', facecolor='none')
    #     ax.add_patch(rect)

    # # 无论是否需要保存，都先保存，然后决定是否显示
    # if output_path:
    #     plt.savefig(output_path, dpi=dpi_factor * fig.dpi)

    # if output_path is None or show_after_save:  # 假设有一个全局变量show_after_save控制是否显示
    #     plt.show()

    draw = ImageDraw.Draw(img)
    for box in boxes:
        left, top, right, bottom = box[0], box[1], box[2], box[3]
        draw.line([(left, top), (left, bottom), (right, bottom),
                       (right, top), (left, top)], width=5, fill='green')
    
    plt.imshow(img)
    # plt.show()

    # 保存预测的图片结果
    if output_path:
        img.save(output_path)


def post_processing(img_path, boxes, name):
    # 读取方框坐标

    # 打开图像
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    height, width = img.shape[-2:]
    
    t_postprocessing_start = time_synchronized()  # 记录后处理开始时间

    # 处理方框数据
    (top_boxes, bottom_boxes) = process_boxes(boxes, width, height)

    all_new_boxes = []

    for sorted_boxes in (top_boxes, bottom_boxes):
        if not sorted_boxes:
            print("方框处理失败，程序退出。")
            exit(1)

        # 计算所有方框的最大宽度的一半
        max_width = calculate_max_width(sorted_boxes)
        epsilon = max_width / 2

        # 提取中心点x坐标并进行DBSCAN聚类
        centers_x = [(box[0] + box[2]) / 2 for box in sorted_boxes]
        labels = perform_dbscan_clustering(centers_x, epsilon, min_samples=1)
        # print("聚类结果:", labels)

        # 获取单独方框簇在sorted_boxes中的索引
        stable_boxes = find_stable_boxes(labels)
        # print("稳定方框索引:", stable_boxes)
        # 获取不重叠的方框在sorted_boxes中的索引
        unintersect_boxes = remove_intersecting_boxes(sorted_boxes)
        # print("不重叠方框索引:", unintersect_boxes)

        # 将列表转换为集合，然后求交集
        intersection_set = set(stable_boxes) & set(unintersect_boxes)
        # 再将交集集合转换回列表
        fixed_boxes = sorted(list(intersection_set))

        # 计算不重叠方框的间距和中心点x坐标，spacings+1=centers
        spacings, centers = calculate_spacing_with_centers(sorted_boxes, fixed_boxes)
        # filtered_spacings:合理间距（去除了离群的长间距）
        # filtered_centers:合理间距前端方框的中心点x坐标
        # filtered_boxes:合理间距头部方框在sorted_boxes中的索引
        # breakpoints:分段的端点的在filtered_spacings/filtered_centers/filtered_boxes中的索引列表
        # 比如只有一段，则是[0,len(filtered_spacings)],两段：[0,x,len(filtered_spacings)],x是变点索引
        filtered_spacings, filtered_centers, filtered_boxes, breakpoints, index_to_remove, fixed_breakpoints = filter_outliers(
            spacings, centers, fixed_boxes)

        # 统计分析和可视化
        # mean_spacing = np.mean(filtered_spacings)
        # median_spacing = np.median(filtered_spacings)
        # print(f"平均间距: {mean_spacing:.2f}")
        # print(f"中位数间距: {median_spacing:.2f}")

        # 可视化统计间距情况
        # visualize_boxes_and_spacings(original_img, sorted_boxes, fixed_boxes, filtered_centers, filtered_spacings)

        # 生成新的方框
        new_boxes = boxes_generator(index_to_remove, fixed_boxes, centers, filtered_boxes, filtered_spacings, labels, breakpoints,
                                    epsilon, centers_x, sorted_boxes, width, filtered_centers, fixed_breakpoints)

        # 可视化每行生成的方框
        # new_original_img = original_img.copy()  # 在第二次绘制前，创建原始图像的一个副本
        # visualize_boxes(new_original_img, new_boxes, output_path=None)

        all_new_boxes.extend(new_boxes)
    
    t_postprocessing_end = time_synchronized()  # 记录后处理结束时间

    # 可视化最终生成的方框（注释掉减少运行时间）
    # visualize_boxes(original_img, all_new_boxes, output_path="./Tunnel-C_test_result/avg_result/" + name + ".jpg")
    # print("最终方框数:", len(all_new_boxes))
    postprocessing_time = t_postprocessing_end - t_postprocessing_start  # 计算后处理时间
    print("后处理时间:", postprocessing_time)

    return all_new_boxes, postprocessing_time
