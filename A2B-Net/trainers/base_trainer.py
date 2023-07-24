# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 15:30:40 2021

@author: wjcongyu
"""
import os
import sys
import glob
from abc import ABC, abstractmethod
import time
class BaseTrainer(ABC):
    def __init__(self, model, config):
        self.model = model
        self.cfg = config 
        self.save_path = ''
        self.checkpoint_file = ''
        
    @abstractmethod
    def start_train(self, train_dataset, val_dataset, save_model_dir):
        '''
        you have to rewrite this function in your sub-class
        '''
        return NotImplemented
    
    def _print_config(self):
        print('======================== configs ========================')
        for key in self.cfg:
            print(key, ':', self.cfg[key])
            
    def _prepare_path(self, save_model_dir):
        print('checking checkpoint path ...')
        if not os.path.exists(self.cfg.CHECKPOINTS_ROOT):
            print('creating checkpoint root:', self.cfg.CHECKPOINTS_ROOT)
            os.mkdir(self.cfg.CHECKPOINTS_ROOT)
            
        model_path = os.path.join(self.cfg.CHECKPOINTS_ROOT, save_model_dir)
        if not os.path.exists(model_path):
            print('creating path for saving weights:', model_path)
            os.mkdir(model_path)
            
        self.save_path = model_path
        
    def _load_pretrained(self, checkpoint_file):
        print('checking pretrained checkpoint file:', checkpoint_file)
        t1 =time.time()
        if not os.path.exists(checkpoint_file):
            checkpoint_file = os.path.join(self.save_path, os.path.basename(checkpoint_file))
            if not os.path.exists(checkpoint_file):
                print('finding the last checkpoint file ...')
                checkpoint_file = self._find_last_checkpoint_file()
                if checkpoint_file == '':
                    print('no checkpoint file found !')
                    return
        
        print('loading weights from:', checkpoint_file)
        self.model.load_weights(checkpoint_file, by_name=True)
        t2 = time.time()
        print('load weights done, used time:', t2-t1)
        
    def _find_last_checkpoint_file(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        weights_files = glob.glob(os.path.join(self.save_path, '*.h5'))
        if len(weights_files) == 0:
            return ''
        weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))
        return weights_files[-1]

    

    def _delete_old_weights(self, nun_max_keep):
        '''
        keep num_max_keep weight files, the olds are deleted
        :param nun_max_keep:
        :return:
        '''
        weights_files = glob.glob(os.path.join(self.save_path, '*.h5'))
        if len(weights_files) <= nun_max_keep:
            return

        weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(x))

        weights_files = weights_files[0:len(weights_files) - nun_max_keep]

        for weight_file in weights_files:
            if 'best' in weight_file: continue

            continued = False
            for epoch in self.cfg.KEEP_CHECKPOINTS:
                exp_checkpoint = "{0}.h5".format(epoch)
                if exp_checkpoint in weight_file:
                    continued = True
                    break
            if continued: continue           

            if weight_file != self.checkpoint_file:
                os.remove(weight_file)

    def _draw_progress_bar(self, cur, total, bar_len=50, maker='='):
        cur_len = int(cur/total*bar_len)
        sys.stdout.write('\r')
        sys.stdout.write("[{:<{}}] {}/{}".format(maker * cur_len, bar_len, cur, total))
        sys.stdout.flush()