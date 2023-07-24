# -*- coding: utf-8 -*-
"""
Created on Fri May  6 10:01:07 2022

@author: wjcongyu
"""

import glob
import cv2
import os.path as osp
import numpy as np
from models.A2BNet import A2BNet
from utils import rbbox as box_manager
from models.backbones import BackboneFactory
from skimage import morphology, feature
from scipy import ndimage

class ChmDetector(object):
    def __init__(self, cfg):
        super(ChmDetector, self).__init__()     
        self.cfg = cfg
        self.model = None
        self.last_error = ''
        
    def initialize(self, checkpoint_file='last', backbone = 'ResNet_34', dtheta=180):
        '''
        call this function before detection on images. build network and load weights
        Input:
            checkpoint_file: the weight file to load
            use_pti: using prior template information or not in the SRGNet
            dtheta: the angle interval for setting rotated anchors
        Output:
            True: initialize done, the detector ready
            False: detector initialize failed
        '''
        try:   
            self.is_ready = False
            print('Start initiazlizing ChromosomeDetector...')
            self.cfg.ANCHOR_DETA_THETA = dtheta
            self.cfg.BACKBONE = backbone
            self.n_anchors = len(self.cfg.ANCHOR_RATIOS)*len(self.cfg.ANCHOR_SCALES) * (180//self.cfg.ANCHOR_DETA_THETA)
            backboneFactory = BackboneFactory()
           
            self.model = A2BNet(backboneFactory.get_backbone(self.cfg.BACKBONE), n_classes=self.cfg.NCLASSES, n_reg_params = 5, n_anchors=self.n_anchors)
            
            self.model.build(input_shape = tuple([self.cfg.BATCH_SIZE]+self.cfg.INPUT_SHAPE + [3]))
            self.model.print_summary()
            #loading checkpoint file specified in the config file. if no checkpoint file found, try to load the latest one.
            
            if not osp.exists(checkpoint_file):
                print('No checkpoint file found:', checkpoint_file)
                
                search_roots = [osp.dirname(checkpoint_file), self.cfg.CHECKPOINTS_ROOT]
                for search_root in search_roots:
                    print('Finding lasted checkpoint file at ', search_root)
                    weights_files = glob.glob(osp.join(search_root, '*.h5'))
                    if len(weights_files) == 0:
                        print('No checkpoint file (h5 format) found!')
                        checkpoint_file = None
                        continue
                    
                    checkpoint_file = sorted(weights_files, key=lambda x: osp.getmtime(x))[-1]
                    print('Found last checkpoint file:', checkpoint_file)
                    break
        
            if checkpoint_file is None:
                return False
            
            print('Loading checkpoints from ', checkpoint_file)
            self.model.load_weights(checkpoint_file)
            
            self.is_ready = True
            print('ChromosomeDetector initializing done !')
            return True                  
                  
        except (OSError, TypeError) as reason:
            self.last_error = str(reason)
            self.is_ready = False
            return False  
        
    def detect(self, image):
        '''
        Input:
            imgs: numpy array of size [N, H, W, C], N is the batch size, C is the channels
            
        Output:
            bboxes: a list with N elements, where each element is a 2D numpy matrix of bboxes                    
                    [[x1, y1, w1, h1, theta1, type1, score1]
                     [x2, y2, w2, h2, theta2, type2, score2]
                     ...
                     [xM, yM, wM, hM, thetaM, typeM, scoreM]]                                                                   
                    where M indicates the number of bbox for a sample image
        '''
                
        assert len(image.shape) == 3 or len(image.shape)==2, 'Image shape must be [H, W] or [H, W, C]'
        assert self.is_ready == True, 'DetNet is not ready, please call <initialize> func fisrt!'
        try:
            #get the preprocessed image for feeding the model: [1, 256, 256, 3]
            oH, oW = image.shape[0:2]
            feed_im = self.__cvt2_feed_format(image.copy())
            nH, nW = feed_im.shape[1:3]            
           
            ske_pred, anchor_cls_pred, anchor_reg_pred = self.model(feed_im, False)   

            stride = nH//anchor_cls_pred.shape[1]

            
            pred_bboxes, masks = self.__get_pred_rbboxes(ske_pred, anchor_cls_pred, anchor_reg_pred, stride)
            if pred_bboxes is None: return None, None
            
            pred_bboxes, masks = self.__cvtbbox2orgspace(pred_bboxes, masks, oH, oW, nH, nW)
           
            return pred_bboxes, masks
        except (OSError, TypeError) as reason:
            self.last_error = str(reason)        
            print('!!!!!!!!!!', self.last_error)
            return None, None

        
    def get_pred_maps(self, image):
        assert len(image.shape) == 3 or len(image.shape)==2, 'Image shape must be [H, W] or [H, W, C]'
        assert self.is_ready == True, 'SGRNet is not ready, please call <initialize> func fisrt!'
        try:
            #get the preprocessed image for feeding the model: [1, 256, 256, 3]
            oH, oW = image.shape[0:2]
            feed_im = self.__cvt2_feed_format(image.copy())
            nH, nW = feed_im.shape[1:3]            
           
            ske_pred, anchor_cls_pred, anchor_reg_pred = self.model(feed_im, False)           
           
            return  ske_pred, anchor_cls_pred, anchor_reg_pred
        except (OSError, TypeError) as reason:
            self.last_error = str(reason)        
            print('!!!!!!!!!!', self.last_error)
            return None, None, None
        
    def __cvt2_feed_format(self, image):
        height, width = image.shape[0:2]
        if height != self.cfg.INPUT_SHAPE[0] or width != self.cfg.INPUT_SHAPE[1]:
            image = cv2.resize(image, (self.cfg.INPUT_SHAPE[1], self.cfg.INPUT_SHAPE[0]))
            
        if len(image.shape)==2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

             
        image = cv2.applyColorMap(np.uint8(image), 2)
       
        image = image.reshape((1, image.shape[0], image.shape[1], -1))
        
        return image.astype('float32')

    
    def __get_pred_rbboxes(self, ske_pred, anchor_cls_pred, anchor_reg_pred, stride):
        
        ske_pred = ske_pred.numpy()
        anchor_cls_pred = anchor_cls_pred.numpy()
        anchor_reg_pred = anchor_reg_pred.numpy()  
        _, H, W, C = anchor_cls_pred.shape
        ske_pred = ske_pred[:,:,:,0].reshape((H, W, 1))
        anchor_cls_pred = anchor_cls_pred.reshape((H, W, C))

        anchor_reg_pred = anchor_reg_pred.reshape((H, W, -1))

        seg_mask = np.argmax(anchor_cls_pred, axis=-1)

        anchor_locs = np.where(seg_mask>0)
        if anchor_locs[0].shape[0]==0: return None, None

        
        anchor_locs = np.hstack([anchor_locs[1][:, np.newaxis], anchor_locs[0][:, np.newaxis]])
       
        anchors = box_manager.generate_ranchors(anchor_locs.copy(), \
                                      self.cfg.ANCHOR_BASE_SIZE, self.cfg.ANCHOR_RATIOS, \
                                      self.cfg.ANCHOR_SCALES, self.cfg.ANCHOR_DETA_THETA, stride)
        
        
       
        xs = anchor_locs[:, 0]; ys = anchor_locs[:, 1]
        
        #default:binary classification, modify here for multi-class detection
        anchor_cls_pred *= ske_pred
        keep_anchor_cls_scores = anchor_cls_pred[ys, xs, ...]  
        
        keep_anchor_reg_detas = anchor_reg_pred[ys, xs, ...]

        keep_anchor_types = np.argmax(keep_anchor_cls_scores, axis=-1)

        keep = np.where((keep_anchor_types>0) & (keep_anchor_types<25))
        keep_anchor_cls_scores = keep_anchor_cls_scores[keep]
        keep_anchor_reg_detas = keep_anchor_reg_detas[keep]
        keep_anchor_types = keep_anchor_types[keep]
        keep_anchors = anchors[keep]
        rois = self.__inverse_bboxes(keep_anchors, keep_anchor_reg_detas)
        
      
        # Clip predicted boxes to image.
        # if y<0 set to 0; elif y>H set to H
        rois[:, 1] = np.clip(rois[:, 1], 0, H*stride)
        # if x<0 set to 0; elif x>W set to W
        rois[:, 0] = np.clip(rois[:, 0], 0, W*stride)
      
        keep_type_scores = keep_anchor_cls_scores[np.arange(keep_anchors.shape[0]), keep_anchor_types]

        if rois is None: return None, None
           
        keep = np.where(keep_type_scores>=self.cfg.OUTPUT_THRES)
        rois = rois[keep]
        if rois.shape[0] == 0: return None, None

        scores = keep_type_scores[keep]
        classes = keep_anchor_types[keep]
        rois = np.hstack([rois, scores[:, None]])
           
       
        keep = box_manager.rotated_nms(rois.astype('float32'),\
                                           iou_thres = self.cfg.NMS_IOU_THRES,\
                                           use_gpu=self.cfg.GPU_NMS)
      
        rois = rois[keep]
        classes = classes[keep]
        rois[:, 2] +=2      
        rois[:, 3] +=2
        
         
        #reassignment of seg_mask
        rois_iou = rois[:, 0:5].copy()           
        rois_iou[:, -1] = -rois_iou[:, -1]/180.0*np.pi    
            
           
        anchors_iou = anchors.copy()
        anchors_iou[:, -1] = -anchors_iou[:, -1]/180.0*np.pi
        ious = box_manager.riou_gpu(rois_iou, anchors_iou)

        deta = 0.0001
        masks = np.zeros((H, W, rois.shape[0]), np.uint8)

        overlaps = box_manager.riou_gpu(rois_iou, rois_iou)
        overlaps[overlaps>=0.99] = 0
        max_overlap = np.max(overlaps, axis=-1)
        for i in range(ious.shape[0]):
            i_ious = ious[i, ...]
            label = classes[i]
            hit_idx = np.where(i_ious>deta)[0]
          
            hit_xs = xs[hit_idx].astype('int16')
           
            hit_ys = ys[hit_idx].astype('int16')
           
            masks[hit_ys, hit_xs, i] = 1
            #contain intersection regions
            #if np.sum(seg_mask[hit_ys, hit_xs]==25)>=3:
            if max_overlap[i]>0.001:
                masks[(seg_mask!=label)&(seg_mask!=25),i] = 0

           
        pred_bboxes = np.hstack([rois, classes[:, None]])
        return pred_bboxes, masks
    
   
    def __inverse_bboxes(self, anchors, anchor_detas): 
        dx, dy, dw, dh, dtheta = np.split(anchor_detas, 5, axis=1)
        
        ax, ay, aw, ah, atheta = np.split(anchors, 5, axis=1)
        
        px = dx*aw+ax; py = dy*ah+ay; pw = np.exp(dw)*aw; ph = np.exp(dh)*ah
        ptheta = dtheta*180.0/np.pi + atheta
      
        pbboxes = np.squeeze(np.stack((px, py, pw, ph, ptheta))).transpose(1,0)
       
        return pbboxes
    
    def __cvtbbox2orgspace(self, pred_bboxes, masks, oH, oW, nH, nW):
        dx = oW/nW
        dy = oH/nH
        
        pred_bboxes[:, 0] *= dx
        pred_bboxes[:, 1] *= dy
        pred_bboxes[:, 2] *= dx
        pred_bboxes[:, 3] *= dy
        
        masks = cv2.resize(masks, (oW, oH))
        

        dilatation_size = 5
        dilatation_type = cv2.MORPH_CROSS
        element = cv2.getStructuringElement(
                        dilatation_type,
                        (2*dilatation_size + 1, 2*dilatation_size+1),
                        (dilatation_size, dilatation_size))
        blur_level = 15
        for i in range(masks.shape[-1]):
            mask = masks[:,:,i]
            box = pred_bboxes[i]
            x, y, w, h = box[0:4]
            w = max(w, h)*0.5
            x1 = int(max(0, x-w))
            x2 = int(min(oW, x+w))
            y1 = int(max(0, y-w))
            y2 = int(min(oH, y+w))

            mask = mask[y1:y2, x1:x2]

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, element)#cv2.dilate(mask, element)
            mask = cv2.blur(mask, (blur_level, blur_level))
            # apply smoothing to the mask
            masks[y1:y2,x1:x2,i] = ndimage.binary_fill_holes(mask)

           
        masks[masks>0] = 1
        return pred_bboxes, masks
    
    