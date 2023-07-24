# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 21:36:13 2021

@author: wjcongyu
"""
import cv2
import numpy as np
#from .rotation.rbbox_overlaps import rbbx_overlaps
from .rotation.rotate_cython_nms import rotate_cpu_nms
from .rotation.rotate_polygon_nms import rotate_gpu_nms
from .rotation.rotate_iou import rotate_iou_gpu_eval
def generate_dense_anchor_loc(width, height, stride=1, mask=None):   
    x = np.arange(0, width, stride)
    y = np.arange(0, height, stride)
    xs, ys = np.meshgrid(x, y, indexing='ij')
    xs = xs.reshape(-1, 1); ys = ys.reshape(-1, 1)
    if mask is not None:
        maskt = np.zeros_like(mask)
        maskt[ys, xs] = 1
        maskt = maskt*mask
        keep_locs = np.where(maskt==1)
        keep_locs = np.hstack([keep_locs[1][:, np.newaxis], keep_locs[0][:, np.newaxis]])
        return keep_locs
    else:
        return np.hstack([xs, ys])
    
def generate_ranchors(anchor_locs, base_size = 6, ratios=[2.0, 4.0], scales = [2, 4, 6], dtheta = 15, stride= 2, to_cv2=True):
    '''
    Generate the rotated anchors for each anchor location.
    
    '''
   
    thetas = np.array([k for k in range(0, 180, dtheta)])
    n_ratios = len(ratios); n_scales = len(scales); n_thetas = len(thetas)
    
    base_anchors = np.zeros((n_ratios*n_scales*n_thetas, 5), dtype=np.float32)
    
    for i in range(n_ratios):
        for j in range(n_scales):
            w = base_size * scales[j] * np.sqrt(ratios[i])
            h = base_size * scales[j] * np.sqrt(1. / ratios[i])
            
            for k in range(n_thetas):         
                
                index = i * n_scales * n_thetas + j * n_thetas + k
                base_anchors[index] = (0, 0, w, h, thetas[k])
                
    
    n_locs = anchor_locs.shape[0]
    n_base_anchors = base_anchors.shape[0]
    anchor_locs = np.repeat(anchor_locs, n_base_anchors, axis=0)
    
    base_anchors = np.tile(base_anchors, (n_locs,1))
    
    anchors = np.hstack([(base_anchors[:, 0:2] + anchor_locs)*stride, base_anchors[:,2:]])
    if to_cv2: anchors = cvt_to_cv2v45_rbboxs(anchors)
    return anchors

    
def cvt_to_cv2v45_rbboxs(rbboxs):
       
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

def riou_gpu(boxes1, boxes2):
    return rotate_iou_gpu_eval(boxes1, boxes2)

def riou_cv2(boxes1, boxes2):
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]
    ious = []
    for i, box1 in enumerate(boxes1):
        temp_ious = []
        r1 = ((box1[0], box1[1]), (box1[2], box1[3]), box1[4])
        for j, box2 in enumerate(boxes2):
            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), box2[4])

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                temp_ious.append(inter)
            else:
                temp_ious.append(0.0)
        ious.append(temp_ious)
    return np.array(ious, dtype=np.float32)

   
    
def rotated_nms(bboxes, iou_thres = 0.1, use_gpu=True):
    '''
    Non-maximum suppression for removing overlaped rotated-bboxes with 
    center format [x, y, w, h, theta, score]
    Input: 
        bboxes: the predicted bbox
        
    '''
    
    assert bboxes.shape[1]>=6, 'no score given in the box at index [5]'  
    if use_gpu:
        return rotate_gpu_nms(bboxes, iou_thres, int(0))
    else:
        return rotate_cpu_nms(bboxes, iou_thres)