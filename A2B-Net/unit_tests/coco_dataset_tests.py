import _init_pathes
import os
import numpy as np
import os.path as osp
import imageio
from datasets.coco_dataset import CocoDataset
import datasets.transforms as tsf 
import time

annotation_file = r'D:\data\chromosome\chromosome_24type_det\self_labeled\train.json'
transforms = tsf.TransformCompose([ tsf.CropRectOnMaskCentroid(2048, 2330),\
                                    tsf.Resize([448, 512])])
                                    #tsf.ReplaceUpValues(250, 220),\
                                    #tsf.RandomPseudoColor()])
dataset = CocoDataset(annotation_file, transforms)
save_path = r'D:\data\chromosome\chromosome_24type_det\self_labeled\train'
if not osp.exists(save_path):
    os.mkdir(save_path)
    
bbox_file = open(os.path.join(save_path, 'rbboxes.txt'), 'w', encoding='utf-8')
for idx in range(len(dataset)):
    t1 = time.time()
    image, mask, rbbox = dataset[idx]
    image = np.uint8(image)
    im_file = dataset.get_filename(idx)
    t2 = time.time()
    print('time per sample:',t2-t1)
    print(image.shape, mask.shape)
    save_im_file = osp.join(save_path, osp.basename(im_file))
    imageio.imsave(save_im_file, image)
    mask_file = save_im_file.replace('.png', '_mask.npz')
    np.savez(mask_file, mask)
    for box in rbbox:
        x, y, w, h, angle, label = box[0:6]
        #assert w>1 and h>1, 'w and h ==0'
        if w<3 or h<3:
            continue
        write_line = '{0} {1} {2} {3} {4} {5} {6}\n'.format(im_file, \
                          round(x,1), round(y,1), round(w,1), round(h,1), round(angle,1), int(label))
    
        bbox_file.write(write_line)
        
bbox_file.close()
   