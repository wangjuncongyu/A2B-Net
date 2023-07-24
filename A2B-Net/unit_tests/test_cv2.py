# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:26:11 2022

@author: wjcongyu
"""
import _init_pathes
import os
import time
import cv2
import numpy as np


from utils import rbbox as box_manager


from utils import helpers
import imageio
from cfgs.det_cfgs import cfg
from datasets.my_dataset import MyDataset
from datasets.my_data_loader import MyDataLoader
import os.path as osp
save_dir = 'test_cv2'
if not osp.exists(save_dir):
    os.mkdir(save_dir)
    
def cvt_to_cv2_rbboxs(rbboxs):
       
    for idx, rbbox in enumerate(rbboxs):
        x, y, w, h, theta = rbbox[0:5]
       
        if theta < 90:
            rbboxs[idx, 2] = h
            rbboxs[idx, 3] = w
            rbboxs[idx, 4] = 90 - rbboxs[idx, 4]
        else:    
            rbboxs[idx, 4] = 180-rbboxs[idx, 4]
        
    return rbboxs

    
def cvt_to_general_rbboxs(rbboxs):
       
    for idx, rbbox in enumerate(rbboxs):
        x, y, w, h, theta = rbbox[0:5]
       
        if w<h:
            rbboxs[idx, 2] = h
            rbboxs[idx, 3] = w
            rbboxs[idx, 4] = 90 - rbboxs[idx, 4]
        else:    
            rbboxs[idx, 4] = 180-rbboxs[idx, 4]
        
    return rbboxs

image = np.zeros((512,512, 3), np.uint8)

gt_rbboxs1 = np.array([[100, 100, 40, 15, 30, 1],[256, 256, 15, 40, 10, 2]], np.int16)

gt_rbboxs2 = cvt_to_cv2_rbboxs(cvt_to_general_rbboxs(gt_rbboxs1.copy()))
print(gt_rbboxs1)

print(gt_rbboxs2)
image = np.uint8(helpers.draw_rbboxes2image(gt_rbboxs1, image.copy(), is_gt=True, draw_txt = False, font_size=0.1)) 
image = np.uint8(helpers.draw_rbboxes2image(gt_rbboxs2, image.copy(), is_gt=True, draw_txt = False, font_size=0.1)) 
image = np.flip(image, axis=-1)
cv2.imwrite(osp.join(save_dir, 'im_iou.png'), image)
