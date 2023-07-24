# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:41:32 2021

@author: wjcongyu
"""
import cv2
import json
import os.path as osp
import PIL.Image
import numpy as np
from utils import helpers
from labelme import utils as labelme_utils


class LabelmeDataset(object):
    def __init__(self, data_root,  transforms=None):
        self.transforms = transforms
        self.categories = np.arange(24)
        self.json_files = helpers.find_files(data_root, '.json')
       
        print('Found labelme files in json format:', len(self.json_files))

  
    
    def __len__(self):
        return len(self.json_files)
        
    def get_ncategores(self):
        return len(self.categories)
    
    def __getitem__(self, index):
        json_file = self.json_files[index]
      
        image, masks, labels = self.parse_json_file(json_file)
        
        if self.transforms: 
            image, masks, _ = self.transforms(image, masks)
         
        masks = np.where(masks>0.95, 1, 0)
        
        gt_rbboxes, masks = self.filter_masks_and_get_rbboxes(masks, labels)
      
        #gt_rbboxes = self.cvt_to_general_rbboxs(gt_rbboxes)
        return image, masks, gt_rbboxes
    
    def parse_json_file(self, json_file):
        image = None
        segmentations = []
        labels = []
        
        with open(json_file,'r', encoding='UTF-8') as fp:
            json_data = json.load(fp) 
            image = labelme_utils.img_b64_to_arr(json_data['imageData'])
            height, width = image.shape[:2]
            
            for shapes in json_data['shapes']:
                points=shapes['points']
                try:
                    label = int(shapes['label'])
                except:
                    label = -1
                
                mask = self.polygons_to_mask([height,width], points)
                
                labels.append(label)
                segmentations.append(mask) 
            segmentations = np.array(segmentations).transpose((1,2,0))
        return image, segmentations, labels
     
    def get_filename(self, index, with_patient_id=True):
        fullname = self.json_files[index]
        patient = osp.basename(osp.dirname(fullname))
        filename = osp.basename(fullname)
        if with_patient_id:
            return patient+'_'+filename
        else:
            return filename
        
    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=np.uint8)
        return mask
    
    def filter_masks_and_get_rbboxes(self, segmentations, labels):
        gt_rbboxes = []
        kept_masks = []
        bi_mask = np.sum(segmentations, axis=-1)
       
        for i in range(segmentations.shape[-1]):
            if labels[i]<=0 or labels[i]>24:
                continue
            mask = segmentations[:,:, i]
            if np.sum(mask==1)<20:
                continue
            #building bbox:x y w h angle and category label
            gt_rbbox = self.get_rotatedbox_frm_mask(mask)
          
            if gt_rbbox[0]<10 or gt_rbbox[0]>bi_mask.shape[1]-10 or\
                gt_rbbox[1]<10 or gt_rbbox[1]>bi_mask.shape[0]-10 or\
                gt_rbbox[2]<5 or gt_rbbox[3]<5:
                continue
            
            gt_rbbox.append(labels[i])
            gt_rbboxes.append(np.array(gt_rbbox))

            mask[mask>0] = labels[i]
            kept_masks.append(mask)
            
        gt_rbboxes = np.array(gt_rbboxes)
      
        kept_masks = np.array(kept_masks, dtype=np.uint8)
        if len(kept_masks.shape)==2:
            kept_mask = np.expand_dims(kept_mask, axis=0)
       
        return gt_rbboxes, kept_masks.transpose((1,2,0))
                
 
    def get_rotatedbox_frm_mask(self, mask):
        (contours, _) = \
        cv2.findContours((255*mask).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) == 0:
                continue
        rect = cv2.minAreaRect(contours[0])
        rbbox =[rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2]]
        
        return rbbox
   
    
    def cvt_to_general_rbboxs(self, rbboxs):
        '''
        convert opencv rbbox in range [-90, 0) to general rbbox in range [0, 180)
        '''
        for idx, rbbox in enumerate(rbboxs):
            x, y, w, h, theta = rbbox[0:5]
       
            if w > h:
                rbboxs[idx, 2] = h
                rbboxs[idx, 3] = w
                rbboxs[idx, 4] += 90
            else:
                rbboxs[idx, 4] += 180
        
        return rbboxs
        
        
            
            
    
    