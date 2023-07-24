# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 20:40:28 2021

@author: wjcongyu
"""

import _init_pathes
import os
import numpy as np
import os.path as osp
import imageio
from cfgs.det_cfgs import cfg
from datasets.my_dataset import MyDataset
#from utils.helpers import draw_bboxes2image
import datasets.transforms as tsf 
from utils import helpers
image_root = r'D:\data\chromosome\chromosome_24type_det\self_labeled\test_npz_seg_512x512'
annotation_file = r'D:\data\chromosome\chromosome_24type_det\self_labeled\test_npz_seg_512x512\rbboxes.txt'
transforms = tsf.TransformCompose([ tsf.RandomFlip()])
dataset = MyDataset(image_root, annotation_file, None, transforms)
save_path = 'test_my_dataset'
if not osp.exists(save_path):
    os.mkdir(save_path)

for idx in range(len(dataset)):
    image, target, rbbox = dataset[idx]
    
    image = np.uint8(image)
    im_file = dataset.get_filename(idx)
    vis_im = np.uint8(helpers.draw_rbboxes2image(rbbox, image, is_gt=True, draw_txt = True)) 
    imageio.imsave(osp.join(save_path, osp.basename(im_file).replace('.png', '_bbox.png')), vis_im)