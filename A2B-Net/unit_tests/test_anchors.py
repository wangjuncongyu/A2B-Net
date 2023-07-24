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

cfg.TEST_ANN_FILE = r'D:\data\chromosome\chromosome_24type_det\self_labeled\test_npz_seg_512x512\rbboxes.txt'
TEST_DATA_ROOT = osp.dirname(cfg.TEST_ANN_FILE)
test_dataset = MyDataset(TEST_DATA_ROOT, cfg.TEST_ANN_FILE, None)

    
def cvt_to_cv2_rbboxs(rbboxs):
       
    for idx, rbbox in enumerate(rbboxs):
        x, y, w, h, theta = rbbox[0:5]
       
        if theta < 90:
            rbboxs[idx, 2] = h
            rbboxs[idx, 3] = w
            rbboxs[idx, 4] = 90 - rbboxs[idx, 4]
        else:            
            '''rbboxs[idx, 2] = h
            rbboxs[idx, 3] = w'''
            rbboxs[idx, 4] = 180-rbboxs[idx, 4]
        
    return rbboxs

def get_anchors_iou(ske_target, gt_rbboxs):
    anchor_locs = np.where(ske_target>=cfg.ANCHOR_LOC_THRES)
    anchor_locs = np.hstack([anchor_locs[1][:, np.newaxis], anchor_locs[0][:, np.newaxis]])

    anchors = box_manager.generate_ranchors(anchor_locs.copy(),\
                                                    cfg.ANCHOR_BASE_SIZE, \
                                                    cfg.ANCHOR_RATIOS,\
                                                    cfg.ANCHOR_SCALES,\
                                                    cfg.ANCHOR_DETA_THETA,\
                                                    4)   

    #anchors = cvt_to_cv2_rbboxs(anchors)  
    anchors[:, -1] = -anchors[:, -1]/180*3.1415926
    
    
    gt_rbboxs[:, -2] = -gt_rbboxs[:, -2]/180*3.1415926
   
    t1 = time.time()
    ious = box_manager.riou_gpu(anchors, gt_rbboxs[:, 0:5])#.riou_cv2(anchors, gt_rbboxs[:, 0:5])#box_manager.calculate_rbbox_iou(anchors, gt_rbboxs[:,0:5]) 
    t2 =time.time()
    print('iou calc:', t2-t1)
    argmax_ious = ious.argmax(axis=-1)   
    max_ious = ious[np.arange(anchors.shape[0]), argmax_ious]
           
    anchor_reg_rbboxs = gt_rbboxs[argmax_ious]
           

    anchor_cls_labels = anchor_reg_rbboxs[:, -1]


    anchor_cls_labels[max_ious<=cfg.ANCHOR_NEG_IOU_THRES] = 0
    return anchors, anchor_cls_labels


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

image, ske_target, gt_rbboxs =  test_dataset[17] 
colors = ['#C0C0C0', '#87CEFA','#8B4513','#00CD66','#C67171','#000080','#458B74','#8B0000','#B8860B','#8B5A00',\
              '#8B1C62','#8B008B','#8A2BE2','#66CD00','#9F79EE','#5CACEE','#4876FF','#76EE00','#8B3626',\
              '#66CDAA','#008B00','#7F7F7F','#8B8970','#98FB98', '#CD4F39']

save_dir = 'test_anchor'
if not osp.exists(save_dir):
    os.mkdir(save_dir)
'''ske_target = ske_target[:,:, 1]
anchors,  anchor_cls_labels= get_anchors_iou(ske_target, gt_rbboxs)
for i in range(anchors.shape[0]):
    x, y = anchors[i, 0:2]
    label = int(anchor_cls_labels[i])
   
    image[int(y),int(x)] = hex_to_rgb(colors[label])'''

len_weights = np.where(gt_rbboxs[:, 2]>gt_rbboxs[:, 3], \
                                 gt_rbboxs[:, 2]/gt_rbboxs[:, 3], \
                                 gt_rbboxs[:, 3]/gt_rbboxs[:, 2])

print(len_weights)
anchor_weights = 1.0+np.float32(1-1.0/np.exp(len_weights))
print(anchor_weights)
gt_rbboxs[:, 5] = len_weights
'''image = np.uint8(helpers.draw_rbboxes2image(gt_rbboxs, image.copy(), is_gt=True, draw_txt = True, font_size=0.9)) 
image = np.flip(image, axis=-1)
cv2.imwrite(osp.join(save_dir, 'im_iou.png'), image)'''
