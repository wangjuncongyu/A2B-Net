import _init_pathes
import os
import numpy as np
import os.path as osp
import imageio
from datasets.labelme_dataset import LabelmeDataset
from datasets.labelme_dataset import LabelmeDataset
from utils.helpers import draw_rbboxes2image
import datasets.transforms as tsf 
import time

data_root = r'D:\data\chromosome\chromosome_24type_det\self_labeled\test_labelme\211025-003C'

dataset = LabelmeDataset(data_root, tsf.TransformCompose([tsf.Resize([448, 512])]))

save_path = 'test_labelme_dataset'
if not osp.exists(save_path):
    os.mkdir(save_path)

for idx in range(len(dataset)):
    im_file = dataset.get_filename(idx)
    if '129_1_688_378_0.461' not in im_file:
        continue
    t1 = time.time()
    ims, targets, rbboxes = dataset[idx]
    t2 = time.time()
    print('time per sample:',t2-t1)
    print(ims.shape, targets.shape)
    targets = np.max(targets, axis=-1)
    imageio.imsave(osp.join(save_path, osp.basename(im_file).replace('.json', '_mask.png')), np.uint8(targets*255))

    image_box = draw_rbboxes2image(rbboxes, ims, True)
    imageio.imsave(osp.join(save_path, osp.basename(im_file).replace('.json', '_bbox.png')), np.uint8(image_box))
   