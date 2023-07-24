# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 19:45:53 2021

@author: wjcongyu
"""
from easydict import EasyDict as edict
import os.path as osp
cfg = edict()
cfg.DATA_ROOT = r'D:\data\chromosome\ChromSeg-SSL\labeled'
cfg.UNLABEL_DATA_ROOT = r'D:\data\chromosome\ChromSeg-SSL\unlabeled'

cfg.TRAIN_ANNO_FILE = osp.join(cfg.DATA_ROOT, 'train_npz_512x512', 'rbboxes.txt')
cfg.TEST_ANNO_FILE = osp.join(cfg.DATA_ROOT, 'test_npz_512x512', 'rbboxes.txt')

cfg.CHECKPOINTS_ROOT = 'checkpoints'

#training phase
cfg.INPUT_SHAPE = [512,512]
cfg.BACKBONE = 'A2BNet-B'
cfg.NCLASSES = 26 #background + types + intersections

cfg.ANCHOR_LOC_THRES = 0.5
cfg.ANCHOR_BASE_SIZE = 3.0
cfg.ANCHOR_RATIOS = [1.0]#the ratio must >=1.0
cfg.ANCHOR_SCALES = [1.0]
cfg.ANCHOR_DETA_THETA = 180
cfg.ANCHOR_NEG_IOU_THRES = 0.000001
cfg.FEATURE_STRIDE = 2


#cfg.ANCHOR_NUM_NEGATIVES = 300
cfg.LR = 0.0001
#these vaues will be automatically changed according to dataset (see train_DetNet.py)
cfg.DECAY_STEPS = 259
cfg.DECAY_RATE = 0.97
cfg.EPOCHS = 100
cfg.STEPS_PER_EPOCH = 518
cfg.BATCH_SIZE =1 #batchsize for labeled images
cfg.UNLABEL_BATCH_SIZE = 1#batchsize for unlabeled images
cfg.PSEUDOLABEL_WARMUP = 0# using unlabeled trainig if current epoch>=5
cfg.MAX_KEEPS_CHECKPOINTS = 1
cfg.KEEP_CHECKPOINTS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

#testing phase
cfg.OUTPUT_THRES = 0.01
cfg.NMS_IOU_THRES = 0.25
cfg.GPU_NMS = True #change to False if GPU not accessed








