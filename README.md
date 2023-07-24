# A2B-Net
# Code and dataset for sermi-supervised chromosme instance segmentation in metaphase cell images

> This is a single-anchor-single-stage detector for chromosome instance segmentation

### my enviroment
- Winows 10
- Anaconda python 3.7.3
- Tensorflow 2.10.0 with gpu
- cuda 11.6
- pytorch 1.12.0.dev20220504+cu116 (required for building the rotation libs, see utils/how to build rotated_nms.txt)

## dataset
Data available at the baidu cloud:https://pan.baidu.com/s/1D5CwZIFSgxaSmlSUvZW0rQ
download code(提取码)：**coming soon** 


## demo
``` bash
(1)download checkpoint file from https://pan.baidu.com/s/1AbfS3kQ5tD2IkY1eloy50g      (download code: hlii)
(2)put the whole checkpoints dirctor to the root: A2B-Net
(3)open a cmd
(4)cd A2B-Net
(5) python demo.py
```
## training
``` bash
(1)download dataset from https://pan.baidu.com/s/1D5CwZIFSgxaSmlSUvZW0rQ      (download code: coming soon)
(2)put the dataset to your directory. 
(3)change the training dataset path: cfg.DATA_ROOT (in the file det_cfg.py) to your dataset path 
(4)change the training dataset path: cfg.UNLABEL_DATA_ROOT to your path
(5)run run_train_a2bnet.bat to train the model
```

