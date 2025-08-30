# -*- coding:utf-8 -*-
import cv2
import imutils
import numpy as np
from skimage import measure
import os
import pandas as pd
import copy
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance
from scipy.signal import find_peaks
from skimage.morphology import skeletonize
from skimage.feature import corner_harris, corner_peaks
import math

img_res = 0.708*0.708
pix_res = 1.024*1.024

radius_1 = 100   # 70 微米
radius_2 = 200   # 140 微米
radius_3 = 70
radius_4 = 140

def find_extreme_points(contour):
    max_distance = 0
    pt1 = None
    pt2 = None
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            distance = np.linalg.norm(contour[i] - contour[j])
            if distance > max_distance:
                max_distance = distance
                pt1, pt2 = contour[i][0], contour[j][0]
    return pt1, pt2

def calculate_angle(pt1, pt2):
    angle = math.degrees(math.atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
    return angle
def extract_vessel_contours(vessel_image):
    # 假设vessel_image是二值化图像，提取轮廓
    contours, _ = cv2.findContours(vessel_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def calculate_local_vessel_density(vessel_image, target_contour, radius=50):
    # 获取目标轮廓的质心
    moments = cv2.moments(target_contour)
    if moments["m00"] != 0:
        cX = int(moments["m10"] / moments["m00"])
        cY = int(moments["m01"] / moments["m00"])
    else:
        return 0

    # 创建一个圆形掩膜，表示局部区域
    mask = np.zeros_like(vessel_image)
    cv2.circle(mask, (cX, cY), radius, 255, -1)

    # 提取局部区域内的血管轮廓
    local_vessel_image = cv2.bitwise_and(vessel_image, vessel_image, mask=mask)
    local_contours = extract_vessel_contours(local_vessel_image)

    # 计算局部区域内的血管数量
    local_vessel_count = len(local_contours)

    return local_vessel_count

def remove_small_points(image, threshold_point):
    img = image
    img_label, num = measure.label(img, connectivity=2, return_num=True)
    props = measure.regionprops(img_label)

    resMatrix = np.zeros(img_label.shape)
    for i in range(1, len(props)):
        if props[i].area > threshold_point:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp
    resMatrix *= 255
    return resMatrix


def box_count(img, box_size):
    (height, width) = img.shape
    count = 0
    for y in range(0, height, box_size):
        for x in range(0, width, box_size):
            if np.sum(img[y:y + box_size, x:x + box_size]) > 0:
                count += 1
    return count

def fractal_dimension(vessel_image):
    # 二值化图像
    thresholded = vessel_image > 127

    sizes = np.array([2 ** i for i in range(1, int(np.log2(min(vessel_image.shape))) + 1)])

    counts = []
    for size in sizes:
        counts.append(box_count(thresholded, size))

    log_counts = np.log(counts)
    log_sizes = np.log(1 / sizes)

    # 线性回归拟合
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dimension = -coeffs[0]

    return fractal_dimension, log_sizes, log_counts

# 提取中心线
def extract_skeleton(binary_image):
    skeleton = cv2.ximgproc.thinning(binary_image)
    return skeleton

# 计算血管密度
def calculate_vessel_density(contours, image_area):
    total_vessel_length = sum(cv2.arcLength(contour, True) for contour in contours)
    density = total_vessel_length / image_area
    return density

# 计算最近邻距离
def calculate_nearest_neighbor_distances(contours):
    centroids = np.array([np.mean(contour[:, 0, :], axis=0) for contour in contours])
    dist_matrix = distance.cdist(centroids, centroids, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_distances = np.min(dist_matrix, axis=1)
    return nearest_distances

# 计算方向分布
def calculate_direction_distribution(contours):
    directions = []
    for contour in contours:
        if len(contour) >= 2:
            for i in range(len(contour) - 1):
                p1 = contour[i][0]
                p2 = contour[i + 1][0]
                angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
                directions.append(angle)
    return np.array(directions)

# 计算血管分布的统计特征
def calculate_distribution_statistics(contours):
    centroids = np.array([np.mean(contour[:, 0, :], axis=0) for contour in contours])
    mean_position = np.mean(centroids, axis=0)
    variance = np.var(centroids, axis=0)
    std_dev = np.std(centroids, axis=0)
    return mean_position, variance, std_dev

def calculate_curvature(contour):
    # 计算一阶导数和二阶导数

    dx = np.gradient(contour[:,0][:,0])
    dy = np.gradient(contour[:,0][:,1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 计算曲率
    curvature = np.abs(dx * ddy - dy * ddx) / np.power(dx**2 + dy**2, 1.5)
    return curvature

def calculate_curvature_derivative(curvature):
    # 计算曲率的导数
    curvature_derivative = np.gradient(curvature)
    return curvature_derivative

def thin_image(binary_image):
    # 细化处理
    skeleton = skeletonize(binary_image // 255)  # 将图像值归一化到[0, 1]
    return skeleton

def detect_branch_points(skeleton):
    # 使用 Harris 角点检测来找到分叉点
    corners = corner_peaks(corner_harris(skeleton), min_distance=1)
    return corners

def count_branch_points(image):
    skeleton = thin_image(image)
    branch_points = detect_branch_points(skeleton)
    return len(branch_points)

def calculate_curvature_points(contour):
    # 计算曲率，找到拐点
    curvature = np.zeros(contour.shape[0])

    for i in range(2, contour.shape[0] - 2):
        x1, y1 = contour[i - 2][0]
        x2, y2 = contour[i - 1][0]
        x3, y3 = contour[i][0]
        x4, y4 = contour[i + 1][0]
        x5, y5 = contour[i + 2][0]

        k1 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
        k2 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else np.inf
        k3 = (y4 - y3) / (x4 - x3) if (x4 - x3) != 0 else np.inf
        k4 = (y5 - y4) / (x5 - x4) if (x5 - x4) != 0 else np.inf

        curvature[i] = abs((k4 - k1) / (1 + k1 * k4))

    # 寻找曲率峰值作为拐点
    peaks, _ = find_peaks(curvature, height=0.5 * np.max(curvature))

    return peaks

def calculate_distance(pt1, pt2):
    return np.linalg.norm(np.array(pt1) - np.array(pt2))

def distance1(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1 - y2)**2)

def calculate_angle_points(contour):
    # 计算轮廓上每个点的角度
    angles = []
    for i in range(len(contour)):
        prev = contour[i - 1][0] if i > 0 else contour[-1][0]
        current = contour[i][0]
        next = contour[(i + 1) % len(contour)][0]

        vec1 = prev - current
        vec2 = next - current

        angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        angles.append(np.degrees(angle))

    # 寻找角度变化大的点作为拐点
    angle_threshold = 30  # 可调整的角度阈值
    angle_changes = np.abs(np.diff(angles + angles[:1]))
    peaks, _ = find_peaks(angle_changes, height=angle_threshold)

    return peaks


def calculate_all(pre_green_img,hole,name,green_with_hole,pre_green_smooth):

    _, pre_green_img = cv2.threshold(
        cv2.cvtColor(pre_green_img.copy(), cv2.COLOR_BGR2GRAY),
        10, 255,
        cv2.THRESH_BINARY)

    _, pre_green_smooth = cv2.threshold(
        cv2.cvtColor(pre_green_smooth.copy(), cv2.COLOR_BGR2GRAY),
        10, 255,
        cv2.THRESH_BINARY)

    _, green_with_hole = cv2.threshold(
        cv2.cvtColor(green_with_hole.copy(), cv2.COLOR_BGR2GRAY),
        10, 255,
        cv2.THRESH_BINARY)

    _, hole = cv2.threshold(
        cv2.cvtColor(hole.copy(), cv2.COLOR_BGR2GRAY),
        10, 255,
        cv2.THRESH_BINARY)
    constant_green_1 = cv2.copyMakeBorder(pre_green_img, radius_1, radius_1, radius_1, radius_1,
                                          borderType=cv2.BORDER_CONSTANT, value=0)
    constant_green_2 = cv2.copyMakeBorder(pre_green_img, radius_2, radius_2, radius_2, radius_2,
                                          borderType=cv2.BORDER_CONSTANT, value=0)

    green_copy_1 = constant_green_1.copy()
    green_copy_2 = constant_green_2.copy()
    whole_vessle_smooth = pre_green_smooth.copy()
    whole_vessle = pre_green_img.copy()
    contours_vessel = cv2.findContours(whole_vessle,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours_vessel = imutils.grab_contours(contours_vessel)

    contours_vessel_smooth = cv2.findContours(whole_vessle_smooth, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_vessel_smooth = imutils.grab_contours(contours_vessel_smooth)
    qulv_smoothl = []
    for c2 in contours_vessel_smooth:
        if cv2.arcLength(c2,True)>20:
            qulv_smoothl.append(np.nanmean(calculate_curvature(c2)))
    contours_hole = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_hole = imutils.grab_contours(contours_hole)

    contours_green = cv2.findContours(green_with_hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_green = imutils.grab_contours(contours_green)

    hole_aera_l = []
    hole_zhouchang = []
    vessel_aera = []
    vessel_zhouchang = []
    hole_vess_aratio = []
    hole_vess_zratio = []
    hole_num_l = []
    vessel_zhouchang_aera_ratio = []

    I_hole = []

    num_vess = 0  #总血管数
    whole_vess_area = 0  #总血管面积
    whole_vess_prei = 0 #总血管周长
    num_with_hole = 0   #有孔洞血管的个数
    iii = -1
    curvaturel = []
    curvatureldl = []
    roundnessl = []
    anglel = []
    diameters = []
    angle_differences = []
    centers = []
    curvature_differences = []
    diameters2 = []

    for c in contours_vessel:
        iii = iii + 1
        if cv2.arcLength(c, True)>0 :
            mask_red = np.zeros_like(pre_green_img)
            cv2.drawContours(mask_red, contours_vessel, iii, 255, thickness=-1)
            mask1 = np.zeros_like(pre_green_img)
            mask1[mask_red*hole>0]=1
            hole_aera = np.sum(mask1)*img_res/pix_res
            mask2 = np.zeros_like(pre_green_img)
            mask2[mask_red*pre_green_img>0] = 1
            green_aera = np.sum(mask2)*img_res/pix_res
            if green_aera==0:
                continue
            mask3 = np.zeros_like(pre_green_img)
            mask3[mask_red*whole_vessle>0]=1

            if (green_aera+hole_aera) > 60:
                curvature = calculate_curvature(c)
                curvatureld = calculate_curvature_derivative(curvature)
                curvatureldl.append(np.nanmean(curvatureld))
                curvaturel.append(np.mean(curvature))
                whole_vess_area = whole_vess_area + hole_aera + green_aera
                num_vess = num_vess + 1
                vessel_aera.append(hole_aera + green_aera)
                hole_num = 0
                hole_aera = 0
                h_zhouchang = 0
                jjj = -1
                pt1, pt2 = find_extreme_points(c)
                angle = calculate_angle(pt1, pt2)
                M = cv2.moments(c)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centers.append((cX, cY))
                anglel.append(angle)
                for ch in contours_hole:
                    jjj = jjj + 1
                    mask_hole = np.zeros_like(image_red)
                    cv2.drawContours(mask_hole, contours_hole, jjj, 255, thickness=-1)  # 将孔洞轮廓标注出来
                    if np.sum(mask_red * mask_hole) > 0:
                        hole_aera = hole_aera + np.sum(hole[mask_red * mask_hole > 0]) / 255 * img_res / pix_res
                        h_zhouchang = h_zhouchang + cv2.arcLength(ch, True) * 0.708 / 1.024
                        hole_num = hole_num + 1
                hole_num_l.append(hole_num)
                lll = -1
                vessel_zhouchan = 0
                for cg in contours_green:
                    lll = lll + 1
                    mask_grreen = np.zeros_like(image_red)
                    cv2.drawContours(mask_grreen, contours_green, lll, 255, thickness=-1)  # 将孔洞轮廓标注出来
                    if np.sum(mask_red * mask_grreen) > 0:
                        vessel_zhouchan = vessel_zhouchan + cv2.arcLength(cg, True) * 0.708 / 1.024
                vessel_zhouchang.append(vessel_zhouchan)
                whole_vess_prei = whole_vess_prei + vessel_zhouchan
                vessel_zhouchang_aera_ratio.append(
                    vessel_zhouchan / (hole_aera + green_aera))
                if hole_num > 0:
                    I_hole.append(1)
                    num_with_hole = num_with_hole + 1
                else:
                    I_hole.append(0)
                (x, y), radius = cv2.minEnclosingCircle(c)
                diameter = radius * 2
                diameters.append(diameter)
                x1, y1, w1, h1 = cv2.boundingRect(c)

                distance_transform = cv2.distanceTransform(pre_green_img, cv2.DIST_L2, 5)

                # 找到最大内切圆半径和其中心
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distance_transform,
                                                                   mask=cv2.drawContours(np.zeros_like(pre_green_img),
                                                                                         [c], -1, 255,
                                                                                         cv2.FILLED))
                radius2 = max_val
                diameter2 = radius2 * 2
                diameters2.append(diameter2)

                center_x = int(x1 + 0.5 * w1)
                center_y = int(y1 + 0.5 * h1)


                img_countour_1 = green_copy_1[(center_y):(center_y + 2 * radius_1),
                                 (center_x):(center_x + 2 * radius_1)]
                img_countour_2 = green_copy_2[(center_y):(center_y + 2 * radius_2),
                                 (center_x):(center_x + 2 * radius_2)]


                sum = 1
                peri_ratio = 0
                locdesnity_1 = 0
                locdesnity_2 = 0

                if sum > 0:
                    peri_ratio = peri_ratio + float(w1) / h1

                    for i in range(2 * radius_1):
                        for j in range(2 * radius_1):
                            real_dis1 = 0
                            if distance1(i + center_x - radius_1, j + center_y - radius_1, center_x,
                                        center_y) > radius_1:
                                img_countour_1[i][j] = 0

                    countourofthisvessl_1, _ = cv2.findContours(img_countour_1, mode=cv2.RETR_TREE,
                                                                method=cv2.CHAIN_APPROX_SIMPLE)

                    for ii in range(2 * radius_2):
                        for jj in range(2 * radius_2):
                            if distance1(ii + center_x - radius_2, jj + center_y - radius_2, center_x,
                                        center_y) > radius_2:
                                img_countour_2[ii][jj] = 0

                    countourofthisvessl_2, _ = cv2.findContours(img_countour_2, mode=cv2.RETR_TREE,
                                                                method=cv2.CHAIN_APPROX_SIMPLE)

                    for ii in range(2 * radius_3):
                        for jj in range(2 * radius_3):
                            real_dis2 = 0
                            if distance1(ii + center_x - radius_3, jj + center_y - radius_3, center_x,
                                        center_y) > radius_3:
                                img_countour_2[ii][jj] = 0

                    countourofthisvessl_3, _ = cv2.findContours(img_countour_2, mode=cv2.RETR_TREE,
                                                                method=cv2.CHAIN_APPROX_SIMPLE)

                    for ii in range(2 * radius_4):
                        for jj in range(2 * radius_4):
                            real_dis2 = 0
                            if distance1(ii + center_x - radius_4, jj + center_y - radius_4, center_x,
                                        center_y) > radius_4:
                                img_countour_2[ii][jj] = 0

                    countourofthisvessl_4, _ = cv2.findContours(img_countour_2, mode=cv2.RETR_TREE,
                                                                method=cv2.CHAIN_APPROX_SIMPLE)


                hole_aera_l.append(hole_aera)
                hole_zhouchang.append(h_zhouchang)
                hole_vess_aratio.append(hole_aera / (green_aera + hole_aera))
                hole_vess_zratio.append(h_zhouchang / vessel_zhouchan)
                round =  4 * 3.14 * np.array(vessel_aera) / np.array(vessel_zhouchang)**2
                roundnessl.append(np.nanmean(round))
                contour_img1 = np.zeros_like(pre_green_img)
                cv2.drawContours(contour_img1, [c], -1, 255, -1)


    for i in range(len(anglel)):
        min_distance = float('inf')
        closest_index = -1
        for j in range(len(anglel)):
            if i != j:
                distance = calculate_distance(centers[i], centers[j])
                if distance < min_distance:
                    min_distance = distance
                    closest_index = j
        if closest_index != -1:
            angle_diff = abs(anglel[i] - anglel[closest_index])
            curvaturel_diff = abs(curvaturel[i]-curvaturel[closest_index])
            angle_diff = min(angle_diff, 360 - angle_diff)
            angle_differences.append(angle_diff)
            curvature_differences.append(curvaturel_diff)

    if num_vess>0:
        dataDF_pre = pd.concat([
                                pd.DataFrame({'0.文件名': [name]}),
                                pd.DataFrame({'5.血管面积': vessel_aera}),
                                pd.DataFrame({'6.血管周长': vessel_zhouchang}),
                                pd.DataFrame({'7.血管周长面积比': vessel_zhouchang_aera_ratio}),
                                pd.DataFrame({'9.孔洞面积': hole_aera_l}),
                                pd.DataFrame({'10.空洞个数': hole_num_l}),
                                pd.DataFrame({'11.空洞周长': hole_zhouchang}),
                                pd.DataFrame({'12.空洞血管面积比': hole_vess_aratio}),
                                pd.DataFrame({'13.孔洞血管周长比': hole_vess_zratio}),
                                pd.DataFrame({'14.是否有孔洞': I_hole})
                                ],
                               axis=1)

        d3 = dataDF_pre
        dataDF_pre3 = pd.DataFrame()
        dataDF_pre3 = pd.concat([dataDF_pre3, d3], axis=0, ignore_index=True)
        if dataDF_pre3.shape[0]==0:
            data_3 = pd.DataFrame()
        else:
            vesselshu = dataDF_pre3.shape[0]
            zong_vessel_aera = np.sum(dataDF_pre3["5.血管面积"])
            ave_vessel_aera = zong_vessel_aera / vesselshu
            zong_vessel_circumference = np.sum(dataDF_pre3["6.血管周长"])
            ave_vessel_circumference = zong_vessel_circumference / vesselshu
            ave_vessel_circumference_aera = np.sum(dataDF_pre3["7.血管周长面积比"]) / vesselshu
            zong_hole_aera = np.sum(dataDF_pre3["9.孔洞面积"])
            ave_hole_aera = zong_hole_aera / vesselshu
            zong_hole_circumference = np.sum(dataDF_pre3["11.空洞周长"])
            ave_hole_circumference = zong_hole_circumference / vesselshu
            ave_hole_vessel_aera_ratio = zong_hole_aera / zong_vessel_aera
            ave_hole_vessel_zhoucchang_ratio = zong_hole_circumference / zong_vessel_circumference
            vessel_hole_zhanbi = np.sum(dataDF_pre3['14.是否有孔洞']) / len(dataDF_pre3['14.是否有孔洞'])
            vessel_hole_geshu = np.sum(dataDF_pre3['14.是否有孔洞'])
            data_3 = pd.concat([dataDF_pre3,
                                pd.DataFrame({'16.血管个数': [vesselshu]}),
                                pd.DataFrame({'17.总血管面积': [zong_vessel_aera]}),
                                pd.DataFrame({'18.平均血管面积': [ave_vessel_aera]}),
                                pd.DataFrame({'19.总血管周长': [zong_vessel_circumference]}),
                                pd.DataFrame({'20.平均血管周长': [ave_vessel_circumference]}),
                                pd.DataFrame({'21.平均血管周长面积比': [ave_vessel_circumference_aera]}),
                                pd.DataFrame({'22.总孔洞面积': [zong_hole_aera]}),
                                pd.DataFrame({'23.平均孔洞面积': [ave_hole_aera]}),
                                pd.DataFrame({'24.总孔洞周长': [zong_hole_circumference]}),
                                pd.DataFrame({'25.平均孔洞周长': [ave_hole_circumference]}),
                                pd.DataFrame({'26.平均孔洞血管面积比': [ave_hole_vessel_aera_ratio]}),
                                pd.DataFrame({'27.平均孔洞血管周长比': [ave_hole_vessel_zhoucchang_ratio]}),
                                pd.DataFrame({'47.有孔洞血管个数占比': [vessel_hole_zhanbi]}),
                                pd.DataFrame({'48.有孔洞血管个数': [vessel_hole_geshu]})
                                ],
                               axis=1)
        return data_3
    else:
        dataDF_pre = pd.concat([pd.DataFrame({'文件': [name]}),
                                pd.DataFrame({'1.总血管个数': [num_vess]})],
                               axis=1)
        print("###################################################")
        return dataDF_pre

base_add = "Bourbon_tiqushuju//output_merge5"
data_l = os.listdir(base_add)
data_all1 = pd.DataFrame()
data_all2 = pd.DataFrame()
data_all3 = pd.DataFrame()
data_all4 = pd.DataFrame()
data_all5 = pd.DataFrame()

data11 = pd.DataFrame()
data12 = pd.DataFrame()
data13 = pd.DataFrame()
data14 = pd.DataFrame()
data15 = pd.DataFrame()

data21 = pd.DataFrame()
data22 = pd.DataFrame()
data23 = pd.DataFrame()
data24 = pd.DataFrame()
data25 = pd.DataFrame()

data31 = pd.DataFrame()
data32 = pd.DataFrame()
data33 = pd.DataFrame()
data34 = pd.DataFrame()
data35 = pd.DataFrame()

def convert_to_gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

def detect_edges(gray_image, low_threshold=100, high_threshold=200):
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return edges

def smooth_edges(edges, ksize=(5, 5), sigmaX=1.5):
    blurred_edges = cv2.GaussianBlur(edges, ksize, sigmaX)
    return blurred_edges

def create_edge_mask(blurred_edges, threshold=50):
    _, edge_mask = cv2.threshold(blurred_edges, threshold, 255, cv2.THRESH_BINARY)
    return edge_mask

def apply_smooth_edges(image, edge_mask):
    smoothed_image = cv2.bitwise_and(image, image, mask=edge_mask)
    return smoothed_image

def smooth_image_edges(image):
    gray_image = convert_to_gray(image)

    # 检测边缘
    edges = detect_edges(gray_image)

    # 柔化边缘
    blurred_edges = smooth_edges(edges)

    # 创建柔化边缘的掩膜
    edge_mask = create_edge_mask(blurred_edges)

    # 将柔化的边缘应用到原图像
    smoothed_image = apply_smooth_edges(image, edge_mask)

    return smoothed_image

for base1 in data_l:
    names = os.listdir(base_add + "//" + base1)
    for name in names:
        print(name)
        if "yuan" in name and ".png" in name:
            image_yuan = cv2.imread(base_add + "//" + base1 + "//" + name)
            (image_red, g1, r1) = cv2.split(image_yuan)
            uname = name
        if "green" in name and ".png" in name and "yuan" not in name:
            pre_green_img1 = cv2.imread(base_add + "//" + base1 + "//" + name)
            (b2, g2, r1) = cv2.split(pre_green_img1)
            hole = cv2.merge([b2, r1, r1])  # 单独得到孔洞的图
            pre_green_img = cv2.merge([r1, g2, r1])
            hole2 = cv2.merge([r1, b2, r1])
            swap = pre_green_img.copy()
            swap[hole2 > 0] = 255
            green_with_hole = swap.copy()

    pre_green_img_rouhua = smooth_image_edges(pre_green_img)
    data_all= calculate_all(pre_green_img, hole, uname, green_with_hole,pre_green_img_rouhua)

    data_all1 = pd.concat([data_all1, data_all], axis=0)
    data_all.to_excel(base_add + "//" + base1 + "//" + uname[0:len(uname) - 4] + "all_newww3.xlsx")

data_all1.to_excel("Bourbon_tiqushuju//" + "huizongjieguo//" + "all_newwww4.xlsx")

