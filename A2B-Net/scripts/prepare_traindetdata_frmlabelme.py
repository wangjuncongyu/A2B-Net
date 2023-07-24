
import _init_pathes
import os
import cv2
import numpy as np
import os.path as osp
from datasets.labelme_dataset import LabelmeDataset
import datasets.transforms as tsf 
from skimage import morphology
import time
from cfgs.det_cfgs import cfg

from skimage import morphology

def generate_targets(masks, gt_rbboxs, target_size = [256, 256], sigma=1.0, intersection_label=25):
    masks = cv2.resize(masks, (target_size[1], target_size[0]))
    masks = np.uint8(np.ceil(masks)) 
    binary_mask = masks.copy()
    binary_mask[binary_mask>0] = 1
    binary_mask = np.sum(binary_mask, axis=-1)
    binary_mask = np.where(binary_mask>1, 1, 0)

    #gau_targets = np.zeros((masks.shape[0], masks.shape[1], 2), dtype=np.float32)
    gau_targets = np.zeros((masks.shape[0], masks.shape[1], 1), dtype=np.float32)
    for i in range(masks.shape[-1]):
        mask = masks[:,:, i]    
        label = np.max(mask)     
        mask[mask>0] = label  

        gt_rbbox = gt_rbboxs[i]
        assert int(gt_rbbox[-1])==int(label),'label mismatch!'
        
        mask_skel = mask.copy()
        mask_skel[mask_skel>0] = 1
        
        skel = np.uint8(morphology.skeletonize(mask_skel.copy()))
        gau = __generate_gaussian(skel, sigma)
        gau_targets[:,:,0] = np.where(gau_targets[:,:,0]>gau, gau_targets[:,:,0], gau)

       
    masks = np.max(masks, axis=-1)
    masks[binary_mask>0] = intersection_label
    masks = np.expand_dims(masks, axis=-1)
    targets = np.concatenate([masks, gau_targets], axis=-1)
    return targets

def __generate_gaussian(target, sigma=8):
        
    H, W = target.shape[0:2]
    [meshgrid_y, meshgrid_x] = np.meshgrid(range(0, H), range(0, W), indexing='ij')
        
    pts = np.where(target>0)
   
    if pts[0].shape[0] ==0:
        return np.zeros_like(target, dtype=np.float32)
      
    gt_y = np.reshape(pts[0], [-1, 1, 1])
    gt_x = np.reshape(pts[1], [-1, 1, 1])
    gau = np.exp(-((gt_x - meshgrid_x) ** 2 + (gt_y - meshgrid_y) ** 2)\
             / (2 * sigma ** 2+1e-5))
    
    gau = np.max(gau, axis=0)
    return gau


stride = cfg.FEATURE_STRIDE

anno_root = r'D:\data\chromosome\chromosome_24type_det\self_labeled\AutoKary2022_1600x1600' #change this root to your path
transforms = tsf.TransformCompose([tsf.Resize(cfg.INPUT_SHAPE)])
def start_convert():
    for subset in ['test_labelme', 'train_labelme']:#
        dataset = LabelmeDataset(osp.join(anno_root, subset), transforms)
        save_path = osp.join(anno_root, subset.replace('labelme', 'npz_'+str(cfg.INPUT_SHAPE[0])+'x'+str(cfg.INPUT_SHAPE[1])))
        if not osp.exists(save_path):
            os.mkdir(save_path)
    
        bbox_file = open(os.path.join(save_path, 'rbboxes.txt'), 'w', encoding='utf-8')
        total = len(dataset)
        done = 0
        for idx in range(len(dataset)):
            im_file = dataset.get_filename(idx)
            done+=1
          
            print(done, total, im_file)
            t1 = time.time()
            image, mask, rbbox = dataset[idx]
            image = np.uint8(image)
            im_file = dataset.get_filename(idx)
            im_file = im_file.replace('.json', '.png')
            t2 = time.time()
            patient_id = osp.basename(osp.dirname(im_file))
            save_im_file = osp.join(save_path, patient_id+'_'+osp.basename(im_file))
            cv2.imwrite(save_im_file, image)

            
            targets = generate_targets(mask, rbbox, np.array(cfg.INPUT_SHAPE)//stride, 1.0)
           
            target_file = save_im_file.replace('.png', '_target.npz')
            np.savez(target_file, targets)

           
            for box in rbbox:
                x, y, w, h, angle, label = box[0:6]
                assert label>0, im_file
              
                write_line = '{0} {1} {2} {3} {4} {5} {6}\n'.format(osp.basename(save_im_file), \
                          round(x,1), round(y,1), round(w,1), round(h,1), round(angle,1), int(label))
    
                bbox_file.write(write_line)
        
        bbox_file.close()

if __name__ == '__main__':
    start_convert()