# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 14:55:00 2021

@author: wjcongyu
"""
import cv2
import numpy as np
    
class MyDataLoader(object):
    def __init__(self, dataset):
        self.dataset = dataset       
        self.cur_idx = 0
        self.sample_idxs = np.arange(len(dataset))
        np.random.shuffle(self.sample_idxs)
        
    def next_batch(self, batch_size):
        batch_ims = []; batch_targets =[]; batch_rbboxs = []; 
        
        #reading images (number of batch_size)
        for i in range(batch_size):
            sample_idx = self.sample_idxs[int((self.cur_idx + i) % len(self.sample_idxs))]            
            image, targets, rbbox = self.dataset[sample_idx]
            if len(image.shape) == 2:
                image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_GRAY2BGR)
            batch_ims.append(image)     
            batch_targets.append(targets)            
            batch_rbboxs.append(rbbox)
        
        batch_ims = np.array(batch_ims, dtype=np.float32) 
        batch_targets = np.array(batch_targets, dtype=np.float32)        
        
        self.__cycle_samples(batch_size)
        #print(batch_ims.shape, batch_targets.shape)
        return batch_ims, batch_targets, batch_rbboxs
    
    
    def __cycle_samples(self, batch_size):
        self.cur_idx += batch_size
        if self.cur_idx >= len(self.sample_idxs):
            self.cur_idx = 0
            np.random.shuffle(self.sample_idxs)
            
    
       