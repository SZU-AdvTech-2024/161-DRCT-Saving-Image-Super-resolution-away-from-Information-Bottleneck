import numpy as np
import cv2
from tqdm import tqdm
from . import statics
from numba import jit

def split_normal_and_mask(img):
    normal=img_to_normal(img[:,:,0:3])
    mask=img[:,:,3]
    return normal, mask


def cat_normal_and_mask(normal, mask):
    mask=mask==0
    normal[mask, :] = [0, 0, 1]
    mask = np.expand_dims(~mask, 2)
    img = normal_to_img(normal)
    img = np.concatenate((img, mask), axis=2)
    return img

# img是[0,1],mat是[-1,1]
def img_to_normal(img):
    mat = (img-0.5)*2
    mat = normalize_normals(mat)
    return mat


def normal_to_img(normal):
    normal = normalize_normals(normal)
    img = (normal+1)/2
    return img


def _vectorize_mat_list(mat_normal, width=3):
    H = mat_normal.shape[0]
    W = mat_normal.shape[1]
    return mat_normal.reshape(H*W, width)


def _devectorize_mat_list(list_normal, H, W):
    return list_normal.reshape(H, W, 3)


def _normal_to_depth(mat_normal):
    H = mat_normal.shape[0]
    W = mat_normal.shape[1]
    list_normal = _vectorize_mat_list(mat_normal, 3)
    list_depth = np.dot(list_normal, statics.LIST_LIGHTS.T)
    list_depth = list_depth.reshape(H, W, len(statics.LIST_LIGHTS))
    return list_depth


def _depth_to_normal(list_depth, h, w):
    list_depth = np.dot(list_depth, np.linalg.pinv(statics.LIST_LIGHTS.T))
    mat_normal = _devectorize_mat_list(list_depth, h, w)
    return mat_normal


def _cv2_resize_mat_list(list_, h, w, cv2_interpolation):
    N = list_.shape[2]
    list_down = np.zeros((h, w, N))
    for idx in range(N):
        list_down[:, :, idx] = cv2.resize(list_[:, :, idx], (w, h),
                                          interpolation=cv2_interpolation)
    return list_down


def normal_resize(mat_normal, h, w, cv2_interpolation):
    list_depth = _normal_to_depth(mat_normal)
    list_down_depth = _cv2_resize_mat_list(
        list_depth, h, w, cv2_interpolation)
    mat_normal = _depth_to_normal(list_down_depth, h, w)
    return mat_normal


@jit(nopython=True)
def _mean_fliter(mat_normal, num_size, mask):
    H, W, _ = mat_normal.shape
    r = int(num_size/2)
    out = np.zeros_like(mat_normal)
    for h in range(H):
        for w in range(W):
            if mask[h, w] == 0:
                left = max(0, w-r)
                right = min(W-1, w+r)
                up = max(0, h-r)
                down = min(H-1, h+r)
                mat_normal_part = mat_normal[up:down+1, left:right+1, :]
                mask_part = mask[up:down+1, left:right+1]
                num_part = (mask_part == 0).sum()
                out[h, w, 0] = np.sum(mat_normal_part[:, :, 0]) / num_part
                out[h, w, 1] = np.sum(mat_normal_part[:, :, 1]) / num_part
                out[h, w, 2] = np.sum(mat_normal_part[:, :, 2]) / num_part
    return out


def _fliter_and_norm_normals(mat_normal, num_size, mask):
    mat_normal = _mean_fliter(mat_normal, num_size, mask)
    mat_normal = normalize_normals(mat_normal)
    return mat_normal


def _get_z_normals(mat_normal):
    out=np.zeros_like(mat_normal)
    out[:,:,2]=1
    return out


# def _get_avg_normals(mat_normal, mask):
#     mx = np.ma.masked_array(mat_normal, mask=mask==0)
#     mx2=mx.mean(axis=3)
#     return mx.mean(axis=3)


def cal_normal_angle(normalA, normalB):
    normal_angle = np.sum(normalA*normalB, axis=2)
    normal_angle = np.clip(normal_angle, -1, 1)
    normal_angle = np.arccos(normal_angle)
    return normal_angle


def _get_avg_angle(normalA, normalB, mask):
    angle_map = cal_normal_angle(normalA, normalB)
    angle_map[mask]=0
    # avg_angle=sum(sum(angle_map))/sum(mask==0)
    return np.mean(angle_map)


def _get_z_base_mat(mat):
    mat_z = np.zeros_like(mat)
    mat_z[:, :, 2] = 1
    return mat_z


def _get_rotation_angle(normalA, normalB):
    rotation_angle = cal_normal_angle(normalA, normalB)
    H, W = rotation_angle.shape
    rotation_angle = rotation_angle.reshape(H, W, 1)
    rotation_angle = np.concatenate([rotation_angle]*3, axis=2)
    return rotation_angle


def _get_rotation_axis(normalA, normalB):
    rotation_axis = np.cross(normalA, normalB)
    rotation_axis = normalize_normals(rotation_axis)
    return rotation_axis


def _get_A_to_B_rotation_vector(mat_A, mat_B):
    rotation_angle = _get_rotation_angle(mat_A, mat_B)
    rotation_axis = _get_rotation_axis(mat_A, mat_B)
    rotation_vector = rotation_angle*rotation_axis
    return rotation_vector


def _rotate_normals_by_vector(mat_normal, rotation_vector, mask):
    H, W = mat_normal.shape[:2]
    mat_ret = np.zeros_like(mat_normal)
    for h in range(0, H):
        for w in range(0, W):
            if mask[h, w] == 0:
                normal = mat_normal[h, w].reshape(3, 1)
                rotation_mat, _ = cv2.Rodrigues(rotation_vector[h, w])
                mat_ret[h, w] = np.dot(rotation_mat, normal).reshape(3,)
    return normalize_normals(mat_ret)


def _get_detail_component(mat_normal, mat_shape, mask):
    mat_z = _get_z_base_mat(mat_normal)
    rotation_vector_shape_to_z = _get_A_to_B_rotation_vector(
        mat_shape, mat_z)
    mat_detail = _rotate_normals_by_vector(
        mat_normal, rotation_vector_shape_to_z, mask)
    return mat_detail


def extract_normals(mat_normal, mask, num_times):
    """
    都归一化了
    """
    mat_shape=mat_normal.copy()
    with tqdm(total=num_times, desc='Get shape') as bar:
        for _ in range(num_times):
            mat_shape = _fliter_and_norm_normals(mat_shape, 3, mask)
            bar.update(1)
    mat_detail = _get_detail_component(mat_normal, mat_shape, mask)
    return mat_detail, mat_shape


def extract_normals_th(mat_normal, mask, num_th):
    """
    都归一化了
    """
    max_times = 200
    with tqdm(total=max_times, desc='Get shape th') as bar:
        mat_z_normal = _get_z_normals(mat_normal)
        mat_shape=mat_normal.copy()
        
        start=0

        for _ in range(max_times):
            mat_shape = _fliter_and_norm_normals(mat_shape, 3, mask)
            mat_detail = _get_detail_component(mat_normal, mat_shape, mask)
            bar.update(1)
            # angle=_get_avg_angle(mat_detail, mat_z_normal, mask)
            angle_map = cal_normal_angle(mat_detail, mat_z_normal)
            angle_map[mask]=0
            angle_mean=np.mean(angle_map)
            if start==0:
                start=angle_mean
            ratio=(angle_mean-start)/start
            print('angle: '+str(angle_mean))
            print('ratio: '+str(ratio))
            if ratio > num_th:
                break

    return mat_detail, mat_shape


def normalize_normals(mat_normal):
    """
    对矩阵沿第三个维度进行单位化 GBR RGB
    """
    h, w, c = mat_normal.shape
    mat_ret = np.zeros((h, w, c))
    for k in range(0, c):
        mat_ret[:, :, k] = np.divide(
            mat_normal[:, :, k],
            np.linalg.norm(mat_normal, axis=2),
            out=np.zeros_like(mat_normal[:, :, k]),
            where=np.linalg.norm(mat_normal, axis=2) != 0)
    return mat_ret


def restore_normals(mat_detail, mask, mat_shape):
    mat_z = _get_z_base_mat(mat_shape)
    rotation_vector_z_to_shape = _get_A_to_B_rotation_vector(
        mat_z, mat_shape)
    mat_restored = _rotate_normals_by_vector(
        mat_detail, rotation_vector_z_to_shape, mask)
    return mat_restored
