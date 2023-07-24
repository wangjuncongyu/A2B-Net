from dataclasses import replace
from genericpath import exists
from posixpath import basename
import labelme2coco
import argparse
import json
import os
import os.path as osp
import glob
import shutil
def get_args():
    parser = argparse.ArgumentParser(description='Converting Labelme to coco style!')
    parser.add_argument('-src_path', '-src_path', type=str, default=r'D:\data\chromosome\cls_annotated', help='the path of source labelme files!')
    parser.add_argument('--save_path', '-save_path', type=str, default=r'D:\data\chromosome\chromosome_24type_det\self_labeled', help='the path for saving coco!')
    
    return parser.parse_args()

def update_json_imname(json_file, new_im_path_name = None):
    with open(json_file,'r', encoding='UTF-8') as fp:
        json_data = json.load(fp) 
        json_data['imagePath'] = new_im_path_name
       
            
        with open(json_file, 'w') as r:
            json.dump(json_data, r)

if __name__ == '__main__':
    args = get_args()
    train_split_rate = 0.9
    im_save_path = osp.join(args.save_path, 'images')
    if not osp.exists(im_save_path):
        os.mkdir(im_save_path)

    print('copying files! this may take several minutes!')
    patients = os.listdir(args.src_path)
    for p in patients:
        p_root = osp.join(args.src_path, p)
        json_files = glob.glob(osp.join(p_root, '*.json'))
        for jf in json_files:
            target_file = osp.join(im_save_path, p+'_'+osp.basename(jf))
            if osp.exists(target_file):
                continue
            shutil.copy(jf, target_file)
            
            target_imf = osp.join(im_save_path, p+'_'+osp.basename(jf.replace('.json', '.png')))
            shutil.copy(jf.replace('.json', '.png'), target_imf)
            update_json_imname(target_file, osp.basename(target_imf))

    labelme2coco.convert(im_save_path, args.save_path, train_split_rate)





    


