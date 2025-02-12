# 关于论文DRCT: Saving Image Super-resolution away from Information Bottleneck的复现

## 相关链接
[论文链接](https://arxiv.org/abs/2404.00722) 
[代码链接](https://github.com/ming053l/DRCT)
[作者提供的预训练模型](https://drive.google.com/drive/folders/1QJHdSfo-0eFNb96i8qzMJAPw31u9qZ7U?usp=sharing)

## Environment
- [PyTorch >= 1.7](https://pytorch.org/) **(Recommend **NOT** using torch 1.8 and **1.12** !!! It would cause abnormal performance.)**
- [BasicSR == 1.3.4.9](https://github.com/XPixelGroup/BasicSR/blob/master/INSTALL.md) 
### Installation
```
git clone https://github.com/ming053l/DRCT.git
conda create --name drct python=3.8 -y
conda activate drct
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
cd DRCT
pip install -r requirements.txt
python setup.py develop
```
## How To Inference on your own Dataset?

```
python inference.py --input_dir [input_dir ] --output_dir [input_dir ]  --model_path[model_path]
```


## How To Test

- Refer to `./options/test` for the configuration file of the model to be tested, and prepare the testing data and pretrained model.  
- Then run the following codes (taking `DRCT_SRx4_ImageNet-pretrain.pth` as an example):
```
python drct/test.py -opt options/test/DRCT_SRx4_ImageNet-pretrain.yml
```
The testing results will be saved in the `./results` folder.  

- Refer to `./options/test/DRCT_SRx4_ImageNet-LR.yml` for **inference** without the ground truth image.

**Note that the tile mode is also provided for limited GPU memory when testing. You can modify the specific settings of the tile mode in your custom testing option by referring to `./options/test/DRCT_tile_example.yml`.**

## How To Train
- Refer to `./options/train` for the configuration file of the model to train.
- Preparation of training data can refer to [this page](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md). ImageNet dataset can be downloaded at the [official website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php).
- Validation data can be download at [this page](https://github.com/ChaofWang/Awesome-Super-Resolution/blob/master/dataset.md).
- The training command is like
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=4321 drct/train.py -opt options/train/train_DRCT_SRx2_from_scratch.yml --launcher pytorch
```

The training logs and weights will be saved in the `./experiments` folder.


