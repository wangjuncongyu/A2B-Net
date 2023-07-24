# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:13:04 2022

@author: wjcongyu
"""


import time
import glob
import imageio
import numpy as np
import os.path as osp
from wrappers.chm_detector import ChmDetector
from utils import helpers
from cfgs.det_cfgs import cfg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


    
if __name__ =='__main__':
    cfg.OUTPUT_THRES = 0.5 #filter out dets with score<0.5
    chm_detector = ChmDetector(cfg)
    chm_detector.initialize(checkpoint_file ='checkpoints/a2bnet-L_augm1_ssis_no/weights_best.h5',
                             backbone = 'A2BNet-L', dtheta=180)
    print(chm_detector.last_error)
        
    
    im_files = glob.glob(osp.join('demo_images', '*.png'))
    done = 0
    total = len(im_files)
    for im_file in im_files:        
        print('--------------------------')
        print(done, total, im_file)
        done+=1

        image = imageio.imread(im_file, pilmode='RGB')
        
        t = time.time()
       
        i_pred_bboxs, i_masks = chm_detector.detect(image.copy())     
        if   i_pred_bboxs is None:
            print('!!!!!!!!!!!!!No detection resutls!!!') 
            continue    

        print('%%%%%%%%%i_pred_bboxs:', i_pred_bboxs.shape, i_masks.shape)
      
        print('types:', np.unique(i_pred_bboxs[:, -1]))
        print('total used time:', time.time()-t)
          
        im_masked = np.uint8(helpers.draw_rbboxes2image(i_pred_bboxs, image.copy(), i_masks, is_gt=False, draw_txt = True, font_base_size=0.02)) 
        
    
        plt.subplot(1,2,1)
        plt.title('org image')
        plt.xticks([]),plt.yticks([])
        plt.imshow(image, cmap='gray')

        plt.subplot(1,2,2)
        plt.title('image with mask')
        plt.xticks([]),plt.yticks([])
        plt.imshow(im_masked, cmap='gray')
    
        plt.show()
   