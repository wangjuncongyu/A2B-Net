# -*- coding: utf-8 -*-
'''
@Date          : 2022-04-13 21:41:33
@Author        : wjcongyu
@Contact       : wangjun@zucc.edu.cn
@Copyright     : 2022 ZUCC
@License       : CC BY-NC-SA 4.0
@Last Modified : wjcongyu 2022-04-13 21:41:34
@Des           : None

@Log           : None

'''
#from mim import download
import tensorflow as tf
from tensorflow.keras import layers as KL

class SalencyHead(tf.keras.Model):
    def __init__(self, n_channels=1):
        super(SalencyHead, self).__init__()        
        
        self.x1 = KL.Conv2D(256, (3, 3), dilation_rate = 1, activation = 'relu', padding='same') 
        self.x2 = KL.Conv2D(256, (3, 3), dilation_rate = 1, activation = 'relu', padding='same') 
        self.score = KL.Conv2D(n_channels, (3, 3), padding='same', \
                               kernel_initializer='zeros', activation='sigmoid', use_bias=False)
       
    def call(self, x, training=True):        
        x1 = self.x1(x) 
        x2 = self.x2(x1)
        score = self.score(x2)      
        return score



class AnchorHead(tf.keras.Model):
    def __init__(self, n_params, n_anchors):
        super(AnchorHead, self).__init__() 
       
        self.x1 = KL.Conv2D(256, (3, 3), dilation_rate = 1, activation = 'relu', padding='same') 
        self.x2 = KL.Conv2D(256, (3, 3), dilation_rate = 1, activation = 'relu', padding='same') 
      
        self.logit = KL.Conv2D(n_anchors*n_params, (3, 3), padding='same', \
                               kernel_initializer='zeros', activation=None, use_bias=False)
       
    def call(self, x, saliency_map, training=True):    
        x = KL.Concatenate(axis=-1)([x, saliency_map])
        x1 = self.x1(x)      
        x2 = self.x2(x1)       
        logit = self.logit(x2)
      
        return logit

class SegHead(tf.keras.Model):
    def __init__(self, n_classes):
        super(SegHead, self).__init__() 
       
        self.x1 = KL.Conv2D(256, (3, 3), dilation_rate = 1, activation = 'relu', padding='same') 
        self.x2 = KL.Conv2D(256, (3, 3), dilation_rate = 1, activation = 'relu', padding='same') 
        
        self.logit = KL.Conv2D(n_classes, (3, 3), padding='same', \
                               kernel_initializer='zeros', activation=None, use_bias=False)
       
    def call(self, x, saliency_map, training=True):
        x = KL.Concatenate(axis=-1)([x, saliency_map])
        x1 = self.x1(x)        
        x2 = self.x2(x1)       
        logit = self.logit(x2)
      
        return logit

      
class A2BNet(tf.keras.Model):
    def __init__(self, backbone, n_classes, n_reg_params, n_anchors):
        super(A2BNet, self).__init__()     
        self.n_classes = n_classes    
        self.n_reg_params = n_reg_params  
        self.n_anchors = n_anchors     
        self.backbone = backbone  
      
        self.ske_head = SalencyHead(1)
        self.cls_head = SegHead(n_classes)
        self.reg_head = AnchorHead(n_reg_params, n_anchors) 
      
    def call(self, ims, training=True):
        backbone = self.backbone(ims, training)
        ske = self.ske_head(backbone, training)        
        reg = self.reg_head(backbone, ske, training)
        cls = self.cls_head(backbone, ske, training) 
        _, H, W, _ = cls.shape
        cls = KL.Softmax()(tf.reshape(cls, [-1, H, W, self.n_classes]))    
        reg = tf.reshape(reg, [-1, H, W, self.n_anchors, self.n_reg_params])
        return ske, cls, reg
    
    def print_summary(self):
        print('-------------- Network achitecture --------------') 
        print(self.backbone.summary()); print(self.cls_head.summary());
        print(self.reg_head.summary()); print(self.ske_head.summary());
        print(self.summary()) 


    

   