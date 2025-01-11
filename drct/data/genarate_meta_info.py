from os import path as osp
from PIL import Image

from basicsr.utils import scandir


def generate_meta_info():
    """Generate meta info for DIV2K dataset.
    """

    #meta_info_txt_train = 'drct/data/meta_info/meta_info_PS_Blobby_GT_train.txt'
    meta_info_txt_train = 'drct/data/meta_info/meta_info_PS_Sculpture_GT_train.txt'
    #meta_info_txt_train = 'drct/data/meta_info/meta_info_PS_Blobby_and_Sculpture_GT_train.txt'
    meta_info_txt_val = 'drct/data/meta_info/meta_info_PS_Blobby_and_Sculpture_GT_val.txt'
    meta_info_txt_test = 'drct/data/meta_info/meta_info_PS_Blobby_and_Sculpture_GT_test.txt'

    gt_folder1 = 'D:\\AA_mywork\\Dataset\\Normal\\PS_Blobby'
    gt_folder2 = 'D:\\AA_mywork\\Dataset\\Normal\\PS_Sculpture'

    # 获取所有图片路径并排序
    #img_list1 = sorted(list(scandir(gt_folder1)))
    #img_list2 = sorted(list(scandir(gt_folder2)))   

    img_list1 = sorted(['PS_Blobby/' + entry for entry in scandir(gt_folder1)])
    img_list2 = sorted(['PS_Sculpture/' + entry for entry in scandir(gt_folder2)])
    # 计算分割点
    total_imgs1 = len(img_list1)
    total_imgs2 = len(img_list2)
    train_end1 = int(total_imgs1 * 0.8)
    train_end2 = int(total_imgs2 * 0.8)
    val_end1 = int(total_imgs1 * 0.9)
    val_end2 = int(total_imgs2 * 0.9)
    kip_train=2
    kip_val=9 # 36*36->4*4
    kip_test=1
    # 写入训练集
    with open(meta_info_txt_train, 'w') as f_train:       
        #for img_path in img_list1[:train_end1:kip_train]:
            #f_train.write(f'{img_path} \n')
        for img_path in img_list2[:train_end2:kip_train]:
            f_train.write(f'{img_path} \n')
    
    # 写入验证集
    with open(meta_info_txt_val, 'w') as f_val:
        i=0
        for img_path in img_list1[train_end1:val_end1:kip_val]:
            if(i%kip_val==0):
                f_val.write(f'{img_path} \n')
            i+=1
        i=0
        for img_path in img_list2[train_end2:val_end2:kip_val]:
            if(i%(kip_val*2)==0):
                f_val.write(f'{img_path} \n')
            i=i+1

    # 写入测试集
    with open(meta_info_txt_test, 'w') as f_test:
        for img_path in img_list1[val_end1:]:
            f_test.write(f'{img_path} \n')
        for img_path in img_list2[val_end2:]:
            f_test.write(f'{img_path} \n')

def generate_WPS_meta_info():
    """Generate meta info for DIV2K dataset.
    """
    print('generate_meta_info')
    meta_info_txt_test = './drct/data/meta_info/meta_info_WPS_GT_test.txt'
    meta_info_txt_train = './drct/data/meta_info/meta_info_WPS_GT_train.txt'
    meta_info_txt_val = './drct/data/meta_info/meta_info_WPS_GT_val.txt'

    gt_test = 'D:\\AA_mywork\\Dataset\\WPS+\\gt\\test\\normal'
    gt_val = 'D:\\AA_mywork\\Dataset\\WPS+\\gt\\valid\\normal'
    gt_train = 'D:\\AA_mywork\\Dataset\\WPS+\\gt\\train\\normal'

    # 获取所有图片路径并排序
    img_list_train = sorted(list(scandir(gt_train)))
    img_list_test = sorted(list(scandir(gt_test)))
    img_list_val = sorted(list(scandir(gt_val)))


    with open(meta_info_txt_test, 'w') as f:
        for img_path in img_list_test:
            f.write(f'{img_path} \n')
    with open(meta_info_txt_train, 'w') as f:
        for img_path in img_list_train:
            f.write(f'{img_path} \n')
    with open(meta_info_txt_val, 'w') as f:
        for img_path in img_list_val:
            f.write(f'{img_path} \n')


if __name__ == '__main__':
    generate_meta_info()