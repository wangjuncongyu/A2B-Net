# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 15:41:32 2021

@author: wjcongyu
"""


import numpy as np
import pandas as pd
import os.path as osp
import imageio
import glob
class MyDataset(object):
    def __init__(self, data_root, bbox_file,  transforms=None):
        self.data_root = data_root
        self.bbox_file = bbox_file
        assert osp.exists(self.bbox_file), 'No bbox file found:'+self.bbox_file
     
        self.transforms = transforms
        self.categories = []
      
        self.__load_bbox_file(self.bbox_file)
        
        
    def __len__(self):
        return len(self.file_pairs)
    
        
    def get_ncategores(self):
        return len(self.categories)
    
    def get_filename(self, index):
        return self.file_pairs[index][0]
    
    def __load_bbox_file(self, bbox_file):
        self.bboxes = {}
        self.file_pairs = []
        self.categories = []
        print('Start parsing bbox file... ', bbox_file)
        annos = pd.read_table(bbox_file, sep=' ',header=None)
        for i in annos.index.values:
            filename, x, y, w, h, angle, label = annos.iloc[i, 0:7]
            if w<3 or h<3 or label<=0 or label>=25: continue
        
            img_file = osp.join(self.data_root, filename)
            target_file = img_file.replace('.png', '_target.npz')
            if (not osp.exists(img_file)) or (not osp.exists(target_file)):
                continue
            if filename not in self.bboxes:
                self.bboxes[filename] = []
                self.file_pairs.append([img_file, target_file])
                
            self.bboxes[filename].append([x, y, w, h, angle, label])
            if label not in self.categories: self.categories.append(label)

        return self.bboxes
    
    
    def __getitem__(self, index):
        im_file, target_file = self.file_pairs[index]
        image = imageio.imread(im_file, pilmode='RGB')
       
        target = None
        if not (target_file is None):
            target = np.load(target_file, allow_pickle=True)['arr_0']
            '''if len(target.shape) < 4:
                target = np.expand_dims(target, axis=-1) '''

        rbboxs = None  
        if osp.basename(im_file) in self.bboxes:     
            rbboxs = np.array(self.bboxes[osp.basename(im_file)])

        if self.transforms:
            image, target, rbboxs =  self.transforms(image, target, rbboxs)
       
        return image, target, rbboxs
    
   