# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:32:57 2022

@author: wjcongyu
"""

import argparse
import os
import os.path as osp
from datasets.my_dataset import MyDataset
from datasets.my_data_loader import MyDataLoader
from datasets.my_dataset_unlabeled import MyDatasetUnlabeled
from datasets.my_data_loader_unlabeled import MyDataLoaderUnlabeled
from cfgs.det_cfgs import cfg
import datasets.transforms as tsf
from models.A2BNet import A2BNet
from models.backbones import BackboneFactory
from trainers.Trainer_NoSSL import Trainer_NoSSL
from trainers.Trainer_SSLIS import Trainer_SSLIS
from trainers.Trainer_SSL_MeanTeacher import Trainer_SSL_MT
from trainers.Trainer_SSL_FixMatch import Trainer_SSL_FixMatch
from trainers.Trainer_SSL_CPS import Trainer_SSL_CPS
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def get_args():
    parser = argparse.ArgumentParser(description='Train the a2bnet for chromosome instance segmentation')
    parser.add_argument('--backbone', '-backbone', type=str, default='ResNet_101', help='The backbone for feature extraction,\
                        optional: ResNet_18, ResNet_34, ResNet_50, ResNet_101, and ResNet_152')
    parser.add_argument('--ach_loc_thres', '-ach_loc_thres', type=float, default=0.2, help='The threshold for anchor locations')
    parser.add_argument('--use_dice', '-use_dice', type=int, default=1, help='Use dice loss for anchor classfication: 0 no, 1 yes')
    parser.add_argument('--use_augm', '-use_augm', type=int, default=1, help='Use data augmentation, i.e., randomflip, rotate, color')
    parser.add_argument('--ssl', '-ssl', type=int, default=1, help='0:no ssl, 1:ssi-is, 2:mean teacher, 3:matchfix, 4:cps')   
    parser.add_argument('--load', '-load', type=str, default='weights_best.h5', help='Load model from a .h5 file')
    parser.add_argument('--save_dir', '-save_dir', type=str, default='SGRA', help='Load model from a .h5 file')
    parser.add_argument('--epoch', '-epoch', type=int, default=120, help='Epoch for training')
    parser.add_argument('--lr', '-lr', type=float, default=0.0001, help='Base learning rate for training')
   
    return parser.parse_args()

if __name__ == '__main__':
    
    args = get_args()
    transform_ops = []
    dataset = None
    
    if args.use_augm==1:
        transform_ops.extend([\
            tsf.RandomFlip(), \
            tsf.RandomRotate()])
            
        print('@@@@@@@@@@@@@ using augment!', transform_ops)
   
    print('!!!!!!!!!! training from npz style data!')
    
    h = cfg.INPUT_SHAPE[0]
    w = cfg.INPUT_SHAPE[1]
    
    cfg.TRAIN_ANN_FILE = osp.join(cfg.DATA_ROOT, 'train_npz_'+str(h)+'x'+str(w), 'rbboxes.txt')
    TRAIN_DATA_ROOT = osp.dirname(cfg.TRAIN_ANN_FILE)
    assert osp.exists(cfg.TRAIN_ANN_FILE), 'Please call prepare_traindetdata_frmlabelme.py to convert data style from labelme to npz....'            
        
    train_dataset = MyDataset(TRAIN_DATA_ROOT, cfg.TRAIN_ANN_FILE, tsf.TransformCompose(transform_ops))
    train_dataloader = MyDataLoader(train_dataset)

    cfg.TEST_ANN_FILE = osp.join(cfg.DATA_ROOT, 'test_npz_'+str(h)+'x'+str(w), 'rbboxes.txt')
    TEST_DATA_ROOT = osp.dirname(cfg.TEST_ANN_FILE)
    if not osp.exists(cfg.TEST_ANN_FILE):
        test_dataloader = None      
    else:        
        test_dataset = MyDataset(TEST_DATA_ROOT, cfg.TEST_ANN_FILE, None)
        test_dataloader = MyDataLoader(test_dataset)

    if args.ssl>0:
        unlabeled_dataset = MyDatasetUnlabeled(cfg.UNLABEL_DATA_ROOT, tsf.TransformCompose(transform_ops))
        unlabeled_dataloader = MyDataLoaderUnlabeled(unlabeled_dataset)

    cfg.ANCHOR_DETA_THETA = 180#we only set a single-anchor of size 3x3 at each location
    cfg.EPOCHS = args.epoch
    cfg.LR = args.lr
   
    n_anchors = len(cfg.ANCHOR_RATIOS)*len(cfg.ANCHOR_SCALES) * (180//cfg.ANCHOR_DETA_THETA)  
    print('###########n_anchors:', n_anchors)
   
    cfg.ANCHOR_LOC_THRES = args.ach_loc_thres
    cfg.NCLASSES = 26#train_dataset.get_ncategores() #0 for background, 1-24 for chromosomes
    print('classes:', cfg.NCLASSES)

    n_reg_params = 5#(dx, dy, dw, dh, dtheta)
    cfg.STEPS_PER_EPOCH = len(train_dataset)//cfg.BATCH_SIZE
    cfg.DECAY_STEPS = cfg.STEPS_PER_EPOCH
    backboneFactory = BackboneFactory()
    cfg.BACKBONE = args.backbone
    net = A2BNet(backbone=backboneFactory.get_backbone(cfg.BACKBONE), n_classes=cfg.NCLASSES, n_reg_params = n_reg_params, n_anchors=n_anchors)
    if args.ssl == 0:  
        trainer = Trainer_NoSSL(net, cfg, use_dice = (args.use_dice==1)) 
        trainer.start_train(train_dataloader, test_dataloader, args.save_dir, pretrained_file=args.load)

    elif args.ssl == 1:
        pertubations = tsf.TransformCompose([tsf.RandomBrightness(), tsf.RandomContrast()])
        trainer = Trainer_SSLIS(net, cfg, args.use_dice==1, pertubations)
        trainer.start_train(train_dataloader, test_dataloader, unlabeled_dataloader, args.save_dir, pretrained_file=args.load)
    elif args.ssl == 2:
        pertubations = tsf.TransformCompose([tsf.RandomBrightness(), tsf.RandomContrast()])
        trainer = Trainer_SSL_MT(net, cfg, args.use_dice==1, pertubations)
        trainer.start_train(train_dataloader, test_dataloader, unlabeled_dataloader, args.save_dir, pretrained_file=args.load)
    elif args.ssl == 3:
        pertubations = tsf.TransformCompose([tsf.RandomBrightness(), tsf.RandomContrast()])
        trainer = Trainer_SSL_FixMatch(net, cfg, args.use_dice==1, pertubations)
        trainer.start_train(train_dataloader, test_dataloader, unlabeled_dataloader, args.save_dir, pretrained_file=args.load)
    elif args.ssl == 4:
        pertubations = tsf.TransformCompose([tsf.RandomBrightness(), tsf.RandomContrast()])
        net_t = DetNet(backbone=backboneFactory.get_backbone(cfg.BACKBONE), n_classes=cfg.NCLASSES, n_reg_params = n_reg_params, n_anchors=n_anchors)
        trainer = Trainer_SSL_CPS(net, net_t, cfg, args.use_dice==1, pertubations)
        trainer.start_train(train_dataloader, test_dataloader, unlabeled_dataloader, args.save_dir, pretrained_file=args.load)