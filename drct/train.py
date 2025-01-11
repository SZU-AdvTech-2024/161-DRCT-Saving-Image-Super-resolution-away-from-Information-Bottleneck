# flake8: noqa
import os.path as osp

#import hat.archs
#import hat.data
#import hat.models

from drct.archs import *
from drct.data import *
from drct.models import *

from basicsr.train import train_pipeline
# 训练流程从drct/train开始
#python drct/train.py -opt options/train/train_DRCT_SRx2_from_scratch.yml 
# 
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    # 使用当前目录创建文件，而不是basicSR package所在位置的目录
    train_pipeline(root_path)
