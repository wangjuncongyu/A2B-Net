# -*- coding: utf-8 -*-
"""
Created on Wed May  4 13:26:11 2022

@author: wjcongyu
"""
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from .base_trainer import BaseTrainer
from utils import rbbox as box_manager
from tensorflow.keras import backend as KB
from .Model_Wrapper_EMA import Model_Wrapper
   
class Trainer_SSL_MT(BaseTrainer):
    def __init__(self, model, config, use_dice=True, pertubations = None):
        super(Trainer_SSL_MT, self).__init__(model, config)    
        #if True, using skel_target as the weigts for calculating reg loss 
        self.use_dice = use_dice 
        self.model = model
        self.pertubations = pertubations
       
       
    def start_train(self, train_dataset, test_dataset, unlabeled_dataset, save_model_dir, pretrained_file=None):
        '''
        Input:
            train_dataset: the data loader for training
            val_dataset: the dataset for online evaluation
            save_model_dir: the weights save path is osp.join(self.cfg.CHECKPOINTS_ROOT, save_model_dir)
            pretrain: if specified, then loading the pretrained weights
        '''
        self._print_config()
        self._prepare_path(save_model_dir)                
       
        self.model.build(input_shape = tuple([self.cfg.BATCH_SIZE]+self.cfg.INPUT_SHAPE + [3]))
        self.model.print_summary()

        self.model_wrapper = Model_Wrapper(self.model, 0.99)

        # loading pretrained weights must behind model.build
        if not (pretrained_file is None): self._load_pretrained(pretrained_file)    
        
        print('\n----------------------------  start model training -----------------------------') 
        
        lr_schedule =  tf.keras.optimizers.schedules.ExponentialDecay(self.cfg.LR, self.cfg.DECAY_STEPS,\
                                                                      self.cfg.DECAY_RATE, staircase = True)
        #lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[10*548, 20*548, 30*548], values=[0.0001, 0.00001, 0.000001, 0.0000001])
        optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)
       
        self.summary_writer = tf.summary.create_file_writer(self.save_path)
        with self.summary_writer.as_default(): 
            min_loss = 100000
            for self.epoch in range(self.cfg.EPOCHS):
                print ('\n################################### epoch:'+str(self.epoch+1)+'/'+str(self.cfg.EPOCHS))
               
                #training an epoch
                t1 = time.time()
                agv_ske_loss, agv_seg_ce_loss, agv_seg_dice_loss, agv_reg_loss, agv_pseudo_loss = self.__train_epoch(train_dataset, unlabeled_dataset, optimizer)
                t2 = time.time()
                current_lr = KB.eval(optimizer._decayed_lr('float32'))
                
               
                print ('\nSke loss: %f; Seg ce loss: %f; Seg dice loss: %f; Reg loss: %f; Pseudo loss: %f; Lr: %f; Used time (s): %f' % \
                      (agv_ske_loss, agv_seg_ce_loss, agv_seg_dice_loss, agv_reg_loss, agv_pseudo_loss, current_lr, t2-t1)) 
                         
                tf.summary.scalar('ske loss', agv_ske_loss, step = (self.epoch+1))
                tf.summary.scalar('seg ce oss', agv_seg_ce_loss, step = (self.epoch+1))
                tf.summary.scalar('seg dice oss', agv_seg_dice_loss, step = (self.epoch+1))
                tf.summary.scalar('reg loss', agv_reg_loss, step = (self.epoch+1))
                tf.summary.scalar('pseudo loss', agv_pseudo_loss, step = (self.epoch+1))

                self.checkpoint_file = os.path.join(self.save_path, "weights_epoch{0}.h5".format(self.epoch + 1))            
                print ('Saving weights to %s' % (self.checkpoint_file))
                self.model.save_weights(self.checkpoint_file)
                
                self._delete_old_weights(self.cfg.MAX_KEEPS_CHECKPOINTS) 

                #runing evaluation
                eval_ske_loss, eval_seg_ce_loss, eval_seg_dice_loss, eval_reg_loss = self.__eval_epoch(test_dataset)
                eval_sum_loss = (eval_seg_ce_loss+eval_seg_dice_loss+eval_reg_loss)
                print('\nske loss: %f; seg ce loss: %f; seg dice loss: %f; reg loss: %f' % (eval_ske_loss, eval_seg_ce_loss, eval_seg_dice_loss, eval_reg_loss))
                if min_loss >= eval_sum_loss:
                    min_loss = eval_sum_loss
                    best_checkpoint = os.path.join(self.save_path, "weights_best.h5")
                    if os.path.exists(best_checkpoint): os.remove(best_checkpoint)
                    print ('!!! Saving best weights to %s' % (best_checkpoint))
                    self.model.save_weights(best_checkpoint)
                
        print('\n---------------------------- model training completed ---------------------------')               


            
    def __train_epoch(self, train_dataset, unlabeled_dataset, optimizer):
        '''
        train cfg.STEPS_PER_EPOCH (e.g., 100) in an epoch
        Input:
            train_dataset: the data loader for training
            optimizer: the optimizer for update the model parameters
        Output:
            salient_loss: the average loss of the salient map
            cls_loss: the average loss of the anchor classification net
            reg_loss: the average loss of the anchor regression net
            
        '''
      
        losses = {'ske_loss':[], 'seg_ce_loss':[], 'seg_dice_loss':[], 'reg_loss':[], 'pseudo_loss':[]}
        print('Start optimizating ...')
        for step in range(self.cfg.STEPS_PER_EPOCH):           
            bat_images, bat_targets, bat_gt_rbboxs = train_dataset.next_batch(self.cfg.BATCH_SIZE) 
            
            bat_masks = bat_targets[:,:,:,0]; bat_salency_targets = bat_targets[:,:,:,1] 
            bat_anchor_locs, bat_anchors, bat_anchor_labels, bat_anchor_reg_rbboxs, bat_anchor_weights = \
                                                        self.__generate_batch_anchors(bat_masks, bat_salency_targets, bat_gt_rbboxs, stride=self.cfg.FEATURE_STRIDE)

            bat_unlabeled_images = unlabeled_dataset.next_batch(self.cfg.UNLABEL_BATCH_SIZE) 
           
            with tf.GradientTape(persistent=False) as tape:
                
                feed_bat_ims = np.concatenate((bat_images, bat_unlabeled_images), axis=0)
               
                bat_ske_scores, bat_seg_scores, bat_reg_deltas = self.model(self.__cvt2_feed_format(feed_bat_ims.copy()))                 
                
                ske_loss = self.__calculate_ske_loss(bat_salency_targets, bat_ske_scores[0:self.cfg.BATCH_SIZE,...])
                losses['ske_loss'].append(ske_loss)

                skel_targets = bat_salency_targets
                seg_ce_loss = self.__calculate_seg_ce_loss( bat_masks, bat_seg_scores[0:self.cfg.BATCH_SIZE,...], skel_targets)
                losses['seg_ce_loss'].append(seg_ce_loss)

                seg_dice_loss = self.__calculate_seg_dice_loss(bat_masks, bat_seg_scores[0:self.cfg.BATCH_SIZE,...]) if self.use_dice  else 0.0
                losses['seg_dice_loss'].append(seg_dice_loss)
               
                reg_loss = self.__calculate_reg_loss(bat_anchor_locs,\
                                                     bat_anchors,\
                                                     bat_anchor_labels,\
                                                     bat_anchor_reg_rbboxs,\
                                                     bat_reg_deltas[0:self.cfg.BATCH_SIZE,...],\
                                                     bat_anchor_weights)
                losses['reg_loss'].append(reg_loss)  

                self.model_wrapper.update_teacher()
                if self.epoch >= int(self.cfg.PSEUDOLABEL_WARMUP):
                    self.model_wrapper.apply_teacher()
                  
                    bat_pseudo_ske_scores, bat_pseudo_seg_scores, bat_pseudo_reg_deltas = self.model(self.__cvt2_feed_format(feed_bat_ims))
                    bat_pseudo_ske_scores = tf.stop_gradient(bat_pseudo_ske_scores)
                    bat_pseudo_seg_scores = tf.stop_gradient(bat_pseudo_seg_scores)
                    bat_pseudo_reg_deltas = tf.stop_gradient(bat_pseudo_reg_deltas)
                    self.model_wrapper.restore() 
                

                    pseudo_loss = self.__calculate_pseudo_loss(bat_ske_scores, \
                                                           bat_seg_scores,\
                                                           bat_reg_deltas,\
                                                           bat_pseudo_ske_scores,\
                                                           bat_pseudo_seg_scores,\
                                                           bat_pseudo_reg_deltas)
                else:
                    pseudo_loss = 0.0

                losses['pseudo_loss'].append(pseudo_loss)  

                grad = tape.gradient(ske_loss + 2.0*seg_ce_loss + 4.0*seg_dice_loss+2.0*reg_loss + pseudo_loss, self.model.trainable_variables)
                   
                optimizer.apply_gradients(grads_and_vars=zip(grad, self.model.trainable_variables))
               
                
            self._draw_progress_bar(step+1, self.cfg.STEPS_PER_EPOCH)
                
       
        return tf.reduce_mean(losses['ske_loss']),\
                 tf.reduce_mean(losses['seg_ce_loss']),\
                 tf.reduce_mean(losses['seg_dice_loss']),\
                 tf.reduce_mean(losses['reg_loss']),\
                 tf.reduce_mean(losses['pseudo_loss'])

    def __eval_epoch(self, test_dataset):
        losses = {'ske_loss':[], 'seg_ce_loss':[], 'seg_dice_loss':[], 'reg_loss':[]}
        print('Start evaluating ...')
        for i in range(len(test_dataset.dataset)):
            bat_images, bat_targets, bat_gt_rbboxs = test_dataset.dataset[i]
            bat_masks = bat_targets[:,:,0]; bat_salency_targets = bat_targets[:,:,1] 

            bat_images = np.reshape(bat_images, [1]+list(bat_images.shape)).astype(np.float32)          
            bat_masks = np.reshape(bat_masks, [1]+list(bat_masks.shape)).astype(np.float32)
            bat_salency_targets = np.reshape(bat_salency_targets, [1]+list(bat_salency_targets.shape)).astype(np.float32)
            bat_gt_rbboxs = [bat_gt_rbboxs]

            bat_ske_scores, bat_seg_scores, bat_reg_deltas = self.model(self.__cvt2_feed_format(bat_images))  
            bat_anchor_locs, bat_anchors, bat_anchor_labels, bat_anchor_reg_rbboxs, bat_anchor_weights = \
                                                        self.__generate_batch_anchors(bat_masks, bat_salency_targets, bat_gt_rbboxs, stride=self.cfg.FEATURE_STRIDE)

            ske_loss = self.__calculate_ske_loss(bat_salency_targets, bat_ske_scores)
            losses['ske_loss'].append(ske_loss)

            seg_ce_loss = self.__calculate_seg_ce_loss( bat_masks, bat_seg_scores)
            losses['seg_ce_loss'].append(seg_ce_loss)

            seg_dice_loss = self.__calculate_seg_dice_loss(bat_masks, bat_seg_scores) if self.use_dice  else 0.0
            losses['seg_dice_loss'].append(seg_dice_loss)
               
            bat_anchor_weights[:] =1.0
            reg_loss = self.__calculate_reg_loss(bat_anchor_locs, bat_anchors, bat_anchor_labels, bat_anchor_reg_rbboxs, bat_reg_deltas, bat_anchor_weights)
            losses['reg_loss'].append(reg_loss)   
            
            self._draw_progress_bar(i+1, len(test_dataset.dataset), 50, '+')

        return tf.reduce_mean(losses['ske_loss']), tf.reduce_mean(losses['seg_ce_loss']), tf.reduce_mean(losses['seg_dice_loss']), tf.reduce_mean(losses['reg_loss'])

    def __calculate_pseudo_loss(self, ske_scores, seg_scores, reg_deltas, pseudo_ske_scores, pseudo_seg_scores, pseudo_reg_deltas):       
       
        ske_scores = tf.reshape(ske_scores, (-1, 1))
        pseudo_ske_scores = tf.reshape(pseudo_ske_scores, (-1, 1))

        N, H, W, C = seg_scores.shape
        seg_scores = tf.reshape(seg_scores, (-1, C))
        pseudo_seg_scores = tf.reshape(pseudo_seg_scores, (-1, C))

        N, H, W, A, C = reg_deltas.shape
        reg_deltas = tf.reshape(reg_deltas, (-1, C))
        pseudo_reg_deltas = tf.reshape(pseudo_reg_deltas, (-1, C))

        keep = tf.where(ske_scores>0.05)
        if len(keep)>100:
            ske_scores = tf.gather_nd(ske_scores, keep)
            pseudo_ske_scores = tf.gather_nd(pseudo_ske_scores, keep)
            seg_scores = tf.gather_nd(seg_scores, keep)
            pseudo_seg_scores = tf.gather_nd(pseudo_seg_scores, keep)
            reg_deltas = tf.gather_nd(reg_deltas, keep)
            pseudo_reg_deltas = tf.gather_nd(pseudo_reg_deltas, keep)
       
            student_map = tf.keras.layers.Concatenate(axis=-1)([ske_scores, seg_scores, reg_deltas])
            teacher_map = tf.keras.layers.Concatenate(axis=-1)([pseudo_ske_scores, pseudo_seg_scores, pseudo_reg_deltas])
            loss = tf.keras.losses.MeanSquaredError()(teacher_map, student_map)
            return loss
        else:
            return 0.0
        

    def __calculate_ske_loss(self, ske_target, ske_score):
        ske_target = tf.cast(ske_target, ske_score.dtype)
        ske_target = tf.reshape(ske_target, [-1])
        ske_score = tf.reshape(ske_score, [-1])

        num_pos = tf.reduce_sum(tf.cast(ske_target == 1, tf.float32))
        neg_weights = tf.math.pow(1 - ske_target, 4)
        pos_weights = tf.ones_like(ske_score, dtype=tf.float32)
        weights = tf.where(ske_target == 1, pos_weights, neg_weights)
        inverse_preds = tf.where(ske_target == 1, ske_score, 1 - ske_score)

        loss = tf.math.log(inverse_preds + 1e-5) * tf.math.pow(1 - inverse_preds, 2) * weights
        loss = -tf.reduce_sum(loss)/(num_pos+1e-5)
        return loss

        '''loss = -tf.math.pow(tf.math.abs(ske_target-ske_score), 2)*((1.0-ske_target)*tf.math.log(1.0-ske_score)+\
                ske_target*tf.math.log(ske_score))
        return tf.reduce_mean(loss)*100.0'''
    
    def __calculate_seg_ce_loss(self, seg_labels, seg_scores, skel_target=None):
        '''
        the focal loss for keypoints (skeletons, connections, and intersecations) prediction
        Input:
            salient_targets: the ground-truth labels (i.e., gaussian targets, BxHxWx1)
            salient_preds: the predicted featuremap for the keypoints (BxHxWx3)
        Output:
            the scalar loss of the salient map prediction
        '''
       
        seg_scores = tf.reshape(seg_scores, [-1, self.model.n_classes])
        seg_labels = tf.cast(seg_labels, seg_scores.dtype)
        
        seg_labels = tf.reshape(seg_labels, (-1))
        
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()(seg_labels, seg_scores)#fl.sparse_categorical_focal_loss(seg_labels, seg_scores, gamma=2.0)
        weights = tf.ones((seg_scores.shape[0])) if skel_target is None else tf.reshape(skel_target+1.0, [-1])
       
        ce_loss = tf.reduce_mean(ce_loss*weights)
        return ce_loss

    def __calculate_seg_dice_loss(self, seg_labels, seg_scores):
        '''
        the focal loss for keypoints (skeletons, connections, and intersecations) prediction
        Input:
            salient_targets: the ground-truth labels (i.e., gaussian targets, BxHxWx1)
            salient_preds: the predicted featuremap for the keypoints (BxHxWx3)
        Output:
            the scalar loss of the salient map prediction
        '''
       
        seg_scores = tf.reshape(seg_scores, [-1, self.model.n_classes])
        seg_labels = tf.cast(seg_labels, seg_scores.dtype)
        seg_labels = tf.reshape(seg_labels, [-1])
       
        
        seg_labels = tf.one_hot(tf.cast(seg_labels, tf.uint8), self.model.n_classes)
        
        seg_labels = seg_labels[:, 1:]
        seg_scores = seg_scores[:, 1:]

        inter = tf.reduce_sum(seg_labels*seg_scores, axis=0)
        A = tf.reduce_sum(seg_labels, axis=0)
        B = tf.reduce_sum(seg_scores, axis=0)       
        dice = (2.0*inter+1e-5)/(A+B+1e-5)
        dice_loss = 1.0-tf.reduce_mean(dice)     
        dice_loss = tf.math.log((tf.math.exp(dice_loss) + tf.math.exp(-dice_loss)) / 2.0)
      
        return dice_loss
    
    def __calculate_reg_loss(self, anchor_locs, anchors, anchor_labels, reg_bboxs, reg_deltas, weights):
        '''
        compute the penalized KLD loss for the anchor regression subnetwork
        Input:
            ach_reg_targets: the rbbox regression targets, [BxNxA, 5] with each row of (Gx, Gy, Gw, Gh, Gθ)            
            ach_reg_preds: the anchor regression map, [B, H, W, A, 5]
            ach_locations: numpy of the locations (x,y) with size [BxN, 2] with each row of (x, y)
            anchors: anchors corresponding to the ach_locations, [BxNxA, 5] with each row of (Ax,Ay,Aw,Ah,Aθ)
            ach_weights: the penalized weights for the loss calculation, [BxNxA]
            ach_cls_labels: the category labels for each anchor (0: negative; 1: positive; -1:ignore)
        Output:
            the scalar loss of the anchor regression task
            
        '''          
        ach_locations = anchor_locs[:, [0, 2, 1]]#change to [batch_id, y, x]
        pts = ach_locations.tolist()

        reg_deltas = tf.gather_nd(reg_deltas, pts)
        reg_deltas = tf.reshape(reg_deltas, [-1, self.model.n_reg_params])

        keep = tf.where(anchor_labels!=-1)
        reg_deltas = tf.squeeze(tf.gather(reg_deltas, keep))
        anchors = tf.cast(tf.squeeze(tf.gather(anchors, keep)), tf.float32)
        reg_bboxs = tf.squeeze(tf.gather(reg_bboxs, keep))
        weights = tf.squeeze(tf.gather(weights, keep))
     
        pred_rboxs = self.__rbbox_transform_inv(anchors, reg_deltas)
        #print(pred_rboxs[:, -1])

        KL_loss = self.__KL_loss(pred_rboxs, reg_bboxs)
        KL_loss = tf.reduce_mean(tf.reshape(KL_loss, (keep.shape[0], keep.shape[0])), axis=-1)
       
        return tf.cast(tf.reduce_mean(KL_loss*weights), tf.float32)
       
    def __generate_batch_anchors(self, bat_masks, bat_ske_targets, bat_gt_rbboxs, stride = 2):
        '''
        Generate labes for anchor classification, and generate bbox regression targets
        Inputs:            
            bat_salient_targets: the batch of salient targets used to generate anchor locations, [B, H, W, 3]
            bat_gt_rbboxs: the batch of ground truth  rbboxes, list: {[G1, 8],..., [GB, 8]}
            bat_ach_cls_score: the batch of predicted anchor cls scores, used for sampling negative anchors, [B, H, W, A, 2]
        Output:
            bat_ach_locations: numpy of the locations (x,y) with size [BxN, 3] with each row of (batchid, x, y )
            bat_anchors: numpy of anchors corresponding to the ach_locations, [BxNxA, 5] with each row of (Ax,Ay,Aw,Ah,Aθ)
            bat_ach_cls_labels: the labels for anchor classification, [BxNxA], 0:negative, 1:positive, -1:ignored
            bat_ach_weights: the penalized weights for loss calculation, [BxNxA]
            bat_ach_reg_targets: the rbbox regression targets, [BxNxA, 5] with each row of (Gx, Gy, Gw, Gh, Gθ)
        '''
        bat_anchor_locs = []; bat_anchors = []; bat_anchor_labels = []; bat_anchor_reg_rbboxs = []; bat_anchor_weights = []
        #generate target for each batch
        for i in range(bat_masks.shape[0]):
            mask = bat_masks[i]
            ske_target = bat_ske_targets[i]  
            gt_rbboxs = bat_gt_rbboxs[i]           

            anchor_locs = np.where((ske_target>=self.cfg.ANCHOR_LOC_THRES)&(mask<25))# | (ske_score>=self.cfg.ANCHOR_LOC_THRES))
           
            anchor_locs = np.hstack([anchor_locs[1][:, np.newaxis], anchor_locs[0][:, np.newaxis]])
            ys = anchor_locs[:, 1]; xs = anchor_locs[:, 0]
            anchors = box_manager.generate_ranchors(anchor_locs.copy(),\
                                                    self.cfg.ANCHOR_BASE_SIZE, \
                                                    self.cfg.ANCHOR_RATIOS,\
                                                    self.cfg.ANCHOR_SCALES,\
                                                    self.cfg.ANCHOR_DETA_THETA,\
                                                    stride,\
                                                    True)#True:convert to cv2 (v4.5) format      
            
            batch_id = np.array([i for k in range(anchor_locs.shape[0])])[:, np.newaxis]
            anchor_locs = np.hstack([batch_id, anchor_locs])
            bat_anchor_locs.append(anchor_locs)

            bat_anchors.append(anchors)

            #cuda version iou is based on radial format
            gt_rbboxes_iou = gt_rbboxs[:, 0:5].copy()           
            gt_rbboxes_iou[:, -1] = -gt_rbboxes_iou[:, -1]/180.0*np.pi     
            
           
            anchors_iou = anchors.copy()
            anchors_iou[:, -1] = -anchors_iou[:, -1]/180.0*np.pi
            ious = box_manager.riou_gpu(anchors_iou, gt_rbboxes_iou)
            
            argmax_ious = ious.argmax(axis=-1)   
            max_ious = ious[np.arange(anchors.shape[0]), argmax_ious]
          
            anchor_reg_rbboxs = gt_rbboxs[argmax_ious]
            bat_anchor_reg_rbboxs.append(anchor_reg_rbboxs[:, 0:5])

            anchor_labels = anchor_reg_rbboxs[:, -1]
            anchor_labels[max_ious<=self.cfg.ANCHOR_NEG_IOU_THRES] = -1#-1 ignore
            bat_anchor_labels.append(anchor_labels)

          
            len_weights = np.where(anchor_reg_rbboxs[:, 2]>anchor_reg_rbboxs[:, 3], \
                                 anchor_reg_rbboxs[:, 2]/anchor_reg_rbboxs[:, 3], \
                                 anchor_reg_rbboxs[:, 3]/anchor_reg_rbboxs[:, 2])
            
            gt_ious = box_manager.riou_gpu(gt_rbboxes_iou, gt_rbboxes_iou)
            gt_ious[gt_ious>0.99] = 0
            dense_weights = np.max(gt_ious, axis=-1)[argmax_ious]
         
            anchor_weights = len_weights+dense_weights#1.0+np.float32(1-1.0/np.exp(len_weights))
            #print('max dense weights:', np.min(dense_weights),  np.max(dense_weights))
            bat_anchor_weights.append(anchor_weights.astype('float32'))

        return np.vstack(bat_anchor_locs), np.vstack(bat_anchors), np.hstack(bat_anchor_labels),\
                np.vstack(bat_anchor_reg_rbboxs), np.hstack(bat_anchor_weights)
       
    
    def __KL_loss(self, pred_rboxs, ach_reg_targets, tau=1.0):
        '''
        computing the KLD loss between predicted bboxs and their regression target bboxes
        copy from: https://github.com/yangxue0827/RotationDetectionv
        '''
        pred_rboxs = tf.cast(pred_rboxs, tf.float32)
        ach_reg_targets = tf.cast(ach_reg_targets, tf.float32)       
        sigma1, sigma2, mu1, mu2, mu1_T, mu2_T = self.__get_gaussian_param(pred_rboxs, ach_reg_targets)
        KL_distance = tf.reshape(self.__KL_divergence(mu1, mu2, mu1_T, mu2_T, sigma1, sigma2), [-1, 1])
        KL_distance = tf.maximum(KL_distance, 0.0)
       
        KL_distance = tf.maximum(tf.math.log(KL_distance + 1.), 0.)
        KL_similarity = 1.0 / (KL_distance + tau)        
      
        KL_loss = (1.0 - KL_similarity)
        return KL_loss
    
    def __get_gaussian_param(self, boxes_pred, target_boxes, shrink_ratio=1.):
        '''
        converting predicted bboxes and their regression target bboxes to 2D gaussians
        copy from: https://github.com/yangxue0827/RotationDetection
        '''
        x1, y1, w1, h1, theta1 = tf.unstack(boxes_pred, axis=1)
        x2, y2, w2, h2, theta2 = tf.unstack(target_boxes, axis=1)
        x1 = tf.reshape(x1, [-1, 1])
        y1 = tf.reshape(y1, [-1, 1])
        h1 = tf.reshape(h1, [-1, 1]) * shrink_ratio
        w1 = tf.reshape(w1, [-1, 1]) * shrink_ratio
        theta1 = tf.reshape(theta1, [-1, 1])
        x2 = tf.reshape(x2, [-1, 1])
        y2 = tf.reshape(y2, [-1, 1])
        h2 = tf.reshape(h2, [-1, 1]) * shrink_ratio
        w2 = tf.reshape(w2, [-1, 1]) * shrink_ratio
        theta2 = tf.reshape(theta2, [-1, 1])
        theta1 *= np.pi / 180.0
        theta2 *= np.pi / 180.0

        sigma1_1 = w1 / 2 * tf.cos(theta1) ** 2 + h1 / 2 * tf.sin(theta1) ** 2.0
        sigma1_2 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_3 = w1 / 2 * tf.sin(theta1) * tf.cos(theta1) - h1 / 2 * tf.sin(theta1) * tf.cos(theta1)
        sigma1_4 = w1 / 2 * tf.sin(theta1) ** 2 + h1 / 2 * tf.cos(theta1) ** 2.0
        sigma1 = tf.reshape(tf.concat([sigma1_1, sigma1_2, sigma1_3, sigma1_4], axis=-1), [-1, 2, 2]) + tf.cast(tf.linalg.eye(
            2) * 1e-5, tf.float32)

        sigma2_1 = w2 / 2 * tf.cos(theta2) ** 2 + h2 / 2 * tf.sin(theta2) ** 2
        sigma2_2 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_3 = w2 / 2 * tf.sin(theta2) * tf.cos(theta2) - h2 / 2 * tf.sin(theta2) * tf.cos(theta2)
        sigma2_4 = w2 / 2 * tf.sin(theta2) ** 2 + h2 / 2 * tf.cos(theta2) ** 2
        sigma2 = tf.reshape(tf.concat([sigma2_1, sigma2_2, sigma2_3, sigma2_4], axis=-1), [-1, 2, 2]) + tf.cast(tf.linalg.eye(
            2) * 1e-5,tf.float32)

        mu1 = tf.reshape(tf.concat([x1, y1], axis=-1), [-1, 1, 2])
        mu2 = tf.reshape(tf.concat([x2, y2], axis=-1), [-1, 1, 2])

        mu1_T = tf.reshape(tf.concat([x1, y1], axis=-1), [-1, 2, 1])
        mu2_T = tf.reshape(tf.concat([x2, y2], axis=-1), [-1, 2, 1])
        return sigma1, sigma2, mu1, mu2, mu1_T, mu2_T
    
    def __KL_divergence(self, mu1, mu2, mu1_T, mu2_T, sigma1, sigma2):
        '''
        compute the KLD between two 2D gaussians
        copy from: https://github.com/yangxue0827/RotationDetection
        '''
        sigma1_square = tf.linalg.matmul(sigma1, sigma1)
        sigma2_square = tf.linalg.matmul(sigma2, sigma2)
        item1 = tf.linalg.trace(tf.linalg.matmul(tf.linalg.inv(sigma2_square), sigma1_square))
        item2 = tf.linalg.matmul(tf.linalg.matmul(mu2-mu1, tf.linalg.inv(sigma2_square)), mu2_T-mu1_T)
        item3 = tf.math.log(tf.linalg.det(sigma2_square) / (tf.linalg.det(sigma1_square) + 1e-4))
        return (item1 + item2 + item3 - 2) / 2. 

    
    def __cvt2_feed_format(self, images):
        if len(images.shape) == 3:
            H, W, C = images.shape
            images = images.reshape((1, H, W, C))

        for i in range(images.shape[0]):
            images[i] = cv2.applyColorMap(np.uint8(images[i]), 2)

        return images
    
    
    def __rbbox_transform_inv(self, anchors, deltas):
        dx, dy, dw, dh, dt = tf.unstack(deltas, axis=1)
        ax, ay, aw, ah, at = tf.unstack(anchors, axis=1)
        px = dx * aw + ax; py = dy * ah + ay;
        pw = tf.exp(dw) * aw; ph = tf.exp(dh) * ah
        pt = dt * 180.0 / np.pi + at

        return tf.transpose(tf.stack([px, py, pw, ph, pt]))


    