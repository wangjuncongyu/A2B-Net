# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:44:24 2021

@author: wjcongyu
"""

import os
import cv2
import numpy as np
import os.path as osp
import math
from scipy.stats import multivariate_normal
from skimage import morphology

def find_files(root_path, postfix='.json'):
    found_files = []
    for root, dirs, files in os.walk(root_path):
        found_files.extend([osp.join(root, f) for f in files if postfix in f])

    return found_files

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def  vis_mask(mask):
    colors = ['#DCDCDC', '#87CEFA','#8B4513','#00CD66','#C67171','#000080','#458B74','#8B0000','#B8860B','#8B5A00',\
              '#8B1C62','#8B008B','#8A2BE2','#66CD00','#9F79EE','#5CACEE','#4876FF','#76EE00','#8B3626',\
              '#66CDAA','#008B00','#7F7F7F','#8B8970','#98FB98', '#CD4F39', '#FF0000']
    type2colors = {}
    for i in range(0, 26):
        type2colors[i] = hex_to_rgb(colors[i])

    #绘制mask
    image = np.zeros((mask.shape[0], mask.shape[1], 3))
    for i in range(1, np.max(mask)+1):
        image[mask==i] = type2colors[int(i)]
    
    return image

def draw_rbboxes2image(bboxes, image, masks=None, is_gt = True, draw_txt=True, font = cv2.FONT_HERSHEY_PLAIN, font_base_size=0.01, font_thickness=1):
    '''
    Input: 
        bboxes: the boxes drawn to the image, box fomrat [x, y, w, h, ...]
        image: the image to draw
        with_theta: 'True' means rotated bbox ([x, y, w, h, theta]) else
                     [x, y, w, h]
    Output:
        image: image with bbox
    '''
    colors = ['#DCDCDC', '#87CEFA','#8B4513','#00CD66','#C67171','#000080','#458B74','#8B0000','#B8860B','#8B5A00',\
              '#8B1C62','#8B008B','#8A2BE2','#66CD00','#9F79EE','#5CACEE','#4876FF','#76EE00','#8B3626',\
              '#66CDAA','#008B00','#7F7F7F','#8B8970','#98FB98', '#CD4F39']
    type2colors = {}
    for i in range(0, 25):
        type2colors[i] = hex_to_rgb(colors[i])

    
    if masks is not None:
        #绘制mask
        mask = image.copy()
        for i in range(masks.shape[-1]):
            seg = masks[:,:, i]       
            label = bboxes[i, -1] 
            mask[seg>0] = type2colors[int(label)]
    
        image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)

        #绘制边界
        for i in range(masks.shape[-1]):
            mask = masks[:,:, i].copy()
            mask[mask>0] = 255
            mask = mask.astype('uint8')
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            label = bboxes[i, -1] 
            edge_color = (255,255,255)#type2colors[int(label)]
            cv2.drawContours(image, contours, -1, edge_color , 2)

    for bbox in bboxes:
        if is_gt:
            x, y, w, h, theta, cls = bbox[0:6]
            txt = str(int(cls)) 
        else:
            x, y, w, h, theta, score, cls = bbox[0:7]
            txt = str(int(cls)) +'|'+str(round(score,2))  
           
       
        rect = ((x,y), (w, h), theta)
      
        box = cv2.boxPoints(rect)
        box = np.int0(box) #turn into ints
        image = cv2.drawContours(image,[box], 0, type2colors[int(cls)], 2, lineType = cv2.LINE_AA)
       
        if draw_txt:          
            font_size= font_base_size *np.sqrt(w*h)
            font_size = 3.0 if font_size > 3.0 else font_size
            font_size = 0.5 if font_size < 0.5 else font_size 
            text_size, _ = cv2.getTextSize(txt, font, font_size, font_thickness)
            text_w, text_h = text_size
            image = cv2.rectangle(image, (int(x), int(y)), (int(x) + text_w, int(y) + text_h), type2colors[int(cls)], -1)
            image = cv2.putText(image, txt, (int(x), int(y)+text_h), font, font_size, (255, 255, 255), font_thickness)
    return image

def draw_edge2image(mask, image, color=(255, 0, 0)):
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    return cv2.drawContours(image, contour, -1, color, 1)

def save_boxs2file(file, bboxes):
    with open(file,'w') as f:
        for box in bboxes:
            x, y, w, h, theta, score = box[0:6]
            f.write('{0} {1} {2} {3} {4} {5}\n'.format(x, y, w, h, theta, score))
            
def generate_gaussian_targets(targets, target_labels = [1, 2, 3], sigmas=[1.0, 2.0, 1.0]):
    gaus = []
    assert len(target_labels)== len(sigmas),\
            "length of target_labels and sigmas must be equal!"
    
    #generating gaussians for skeletons, connecting regions, and intersection points
    for i, label in enumerate(target_labels):
        i_target = targets.copy()
        i_target[i_target!=label] = 0
        gaus.append(generate_gaussian(i_target, sigmas[i]*1.0))
    return np.array(gaus).transpose((1, 2, 0))
            

def generate_skeleton_gaussian(masks, C=24, sigma=1.0):
    target = np.zeros_like(masks)
    for i in range(C):
        mask = np.where(masks==(i+1), 1, 0)
        if np.max(mask)==0:
            continue
        skel = np.uint8(morphology.skeletonize(mask.copy()))
     
        gau = generate_gaussian(skel, sigma)
      
        target = np.where(target>gau, target, gau)
        
    return target


def generate_gaussian(target, sigma=8):
        
    H, W = target.shape[0:2]
    [meshgrid_y, meshgrid_x] = np.meshgrid(range(0, H), range(0, W), indexing='ij')
        
    pts = np.where(target>0)
   
    if pts[0].shape[0] ==0:
        return np.zeros_like(target, dtype=np.float32)
      
    gt_y = np.reshape(pts[0], [-1, 1, 1])
    gt_x = np.reshape(pts[1], [-1, 1, 1])
    gau = np.exp(-((gt_x - meshgrid_x) ** 2 + (gt_y - meshgrid_y) ** 2)\
             / (2 * sigma ** 2+1e-5))
    
    gau = np.max(gau, axis=0)
    return gau

def generate_rotated_gaussian_mask(mask_shape, gt_y, gt_x, gt_h, gt_w, gt_theta):
    '''
    Input:
        mask_shape: [Height, Width] the size of the output mask
        gt_y: the y locations of the centers (i.e., the bbox center in the vertical direction) 
        gt_x: the x locations of the centers (i.e., the bbox center in the horizontal direction)
        gt_h: the height of the bboxes
        gt_w: the width of the bboxes
        min_overlap: control the sigma of gaussian. 
                     Larger value leads to smaller gaussian target
    Output:
        the gaussian mask with size of [Height, Width]
        
    '''
    H, W = mask_shape
   
    h = range(0, H)
    w = range(0, W)
    [meshgrid_y, meshgrid_x] = np.meshgrid(h, w, indexing='ij')    
    positions = np.array([meshgrid_x, meshgrid_y])
    positions = positions.transpose((1, 2, 0))
    gt_theta = gt_theta/180*math.pi
    
    gau = np.zeros(mask_shape, dtype = np.float32)
    for i in range(gt_y.shape[0]):  
        R = np.array([[math.cos(gt_theta[i]), -1*math.sin(gt_theta[i])], [math.sin(gt_theta[i]), math.cos(gt_theta[i])]])
      
        cov_matrix_root = np.matmul(np.matmul(R, np.array([[gt_w[i]/2, 0], [0, gt_h[i]/2]])), R.T)

        cov_matrix = np.matmul(cov_matrix_root, cov_matrix_root)
        
        #covinv = np.linalg.inv(cov_matrix)
    
        rv = multivariate_normal([gt_x[i], gt_y[i]], cov_matrix)
       
        i_gau = rv.pdf(positions)
        #print(np.max(i_gau))
        i_gau = (i_gau-np.min(i_gau)) / (np.max(i_gau)-np.min(i_gau))
        gau = np.where(i_gau > gau, i_gau, gau)
   
    return gau

                
def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

#for unit test
if __name__ =='__main__':
    root_path = r'D:\data\chromosome\chromosome_labeme2021'
    found_files = find_files(root_path, '.json')
    print(found_files)
    