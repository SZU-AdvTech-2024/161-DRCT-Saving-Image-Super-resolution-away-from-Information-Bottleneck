import numpy as np
from . import normals

def get_img_mask(img, th=0.05):
    """
    对颜色向量求和, 然后像素和小于th的判定为背景
    """
    np_img = np.array(img)
    np_sum = np_img.sum(axis=2)
    return np_sum < th

def img_to_normal(img):

    mat = (img-0.5)*2
    mat = normals.normalize_normals(mat)
    return mat

def read_normal_pair(img_a, img_b):
    # # 将 img 转换为 float32 类型
    img_a = img_a.astype(np.float32)
    img_b = img_b.astype(np.float32)
    # # 将 img 的值范围从 [0, 255] 映射到 [0, 1]
    img_a /= 255.0
    img_b /= 255.0

    mask_a = get_img_mask(img_a)
    mask_b = get_img_mask(img_b)
    mask = mask_a | mask_b

    # 将 img 的值范围从 [0, 1] 映射到 [-1, 1]
    img_a = img_a * 2 - 1
    img_b = img_b * 2 - 1  

    # 计算每个法向量的范数

    norma = np.linalg.norm(img_a, axis=2, keepdims=True)
    normb = np.linalg.norm(img_b, axis=2, keepdims=True)
    norma[mask] = 1
    normb[mask] = 1   
    # 对每个法向量进行归一化处理
    mat_a = img_a / norma
    mat_b = img_b / normb
    return mat_a, mat_b, mask
# 计算anglemap
def calculate_anglemap(normal_a, normal_b, mask):
    angle_map = normals.cal_normal_angle(normal_a, normal_b)
    angle_map = angle_map * 180 / np.pi
    angle_map[mask] = 0
    return angle_map

# 计算平均angle
def calculate_mean_ae(angle_map, mask):
    return np.mean(angle_map[mask == 0])

# 计算中位
def calculate_median_ae(angle_map, mask):
    return np.median(angle_map[mask == 0])

# 计算角度小于th的比例
def calculate_th_percent(angle_map, mask, th=5):
    return np.sum(angle_map[mask == 0] <= th)/np.sum(mask == 0)


def calculate_angle_variation(angle_map, mask):
    return np.var(angle_map[mask == 0])

def metric_single_img(img_a, img_b):
    mat_a, mat_b, mask = read_normal_pair(img_a, img_b)
    angle_map = calculate_anglemap(mat_a, mat_b, mask)
    mean_ae = calculate_mean_ae(angle_map, mask)
    median_ae = calculate_median_ae(angle_map, mask)
    th_percent_5 = calculate_th_percent(angle_map, mask, 5)
    th_percent_10 = calculate_th_percent(angle_map, mask, 10)
    variation = calculate_angle_variation(angle_map, mask)
    return mean_ae, median_ae, th_percent_5, th_percent_10, variation

