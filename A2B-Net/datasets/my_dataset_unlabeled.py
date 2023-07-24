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

class MyDatasetUnlabeled(object):
    def __init__(self, data_root, transforms=None):
        self.data_root = data_root
       
       
        assert osp.exists(self.data_root), 'No root found:'+self.data_root
        
        self.transforms = transforms
        self.files = glob.glob(osp.join(data_root, '*.png'))
        
    def __len__(self):
        return len(self.files)
    
        
    def get_ncategores(self):
        return 24
    
    def get_filename(self, index):
        return self.files[index]  

    
    def __getitem__(self, index):
        im_file = self.files[index]
        image = imageio.imread(im_file, pilmode='RGB')
       
        if self.transforms:
            image, _, _ =  self.transforms(image)
          
        return image
    
   