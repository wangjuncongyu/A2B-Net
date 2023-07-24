import _init_pathes
import os
import numpy as np
import os.path as osp
import imageio
from datasets.coco_dataset import CocoDataset
from datasets.det_data_loader import DetDataLoader
from utils.helpers import draw_bboxes2image
import datasets.transforms as tsf 
import time

annotation_file = r'D:\data\chromosome\chromosome_24type_det\caisi_labeled\train.json'

dataset = CocoDataset(annotation_file, tsf.TransformCompose([ #tsf.CropRectOnMaskCentroid(2048, 2330),\
                                    tsf.Resize([448, 512])
                                    ]))

dataloader = DetDataLoader(dataset, tsf.TransformCompose([tsf.ReplaceUpValues(250, 220),\
                                    tsf.RandomPseudoColor()]))
save_path = 'test_det_dataloader'
if not osp.exists(save_path):
    os.mkdir(save_path)
for i in range(10):
    t1 = time.time()
    ims, targets, rbboxes = dataloader.next_batch(1)
    t2 = time.time()
    print('time per sample:',t2-t1)
    print(ims.shape, targets.shape)
    imageio.imsave(osp.join(save_path, str(i)+'_target.png'), np.uint8(targets[0,:,:,0]*255))
    image_box = draw_bboxes2image(rbboxes[0], ims[0,...], True)
    imageio.imsave(osp.join(save_path, str(i)+'_im.png'), np.uint8(image_box))
   