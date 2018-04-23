from myutils import *
import h5py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.init
from torch.utils.data import Dataset

class vqaDataset(Dataset):
    """vqa dataset."""
    def __init__(self, opt, mode = 'train'):
        """
        Args:
            qa_dir (string): Path to the h5 file with annotations.
            img_dir (string): Path to the h5 file with image features
            mode (string): Mode of train or test
        """
        self.qa_dir = opt.qa_dir
        self.vocab_dir = opt.vocab_dir
        self.img_h5 = opt.vqa_img_h5
        self.num_output = opt.num_output

        # hidden trick
        self._eye = torch.nn.init.eye(torch.Tensor(self.num_output, self.num_output))

        # qa h5
        file = h5py.File(os.path.join(opt.vocab_dir, 'annotation_vqa.h5'), 'r') 
        self.qas = {}
        self.qas['question'] = torch.from_numpy(np.int64(file['/ques_train'][:]))
        self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
        self.qas['img_id'] = torch.from_numpy(np.int32(file['/img_id_train'][:]))
        self.qas['answers'] = torch.from_numpy(np.int64(file['/answers'][:]))
        file.close()
        # img h5
        self.img_file = h5py.File(self.img_h5, 'r')

        # train_test json
        self.train_test = read_json(os.path.join(opt.vocab_dir, 'vqa_train-val.json'))
        self.cur_list = range(len(self.qas['question_id']))
        self.trees = read_json(os.path.join(opt.vocab_dir, 'vqa_sorted_trees.json'))

        print('    * vqa-%s loaded' % mode)

    def __len__(self):
        return len(self.cur_list)
    
    def __getitem__(self, idx):
        idx = self.cur_list[idx]  
        scene_graph = [] # not yet
        img_id = self.qas['img_id'][idx]
        qid = self.qas['question_id'][idx]
        
        def id2imgName(img_id, qid):
            if self.train_test[str(qid)] == 0:
                return 'COCO_train2014_%012d.jpg' % img_id
            else:
                return 'COCO_val2014_%012d.jpg' % img_id

        def get_answer():
            answer = self.qas['answers'][idx] - 1
            rn = torch.randperm((answer >= 0).sum())[0]
            return answer[rn]
        
        img_name = id2imgName(img_id, qid)

        return self.qas['question'][idx], \
               qid, \
               get_answer(), \
               torch.from_numpy(np.array(self.img_file[img_name])), \
               img_name, \
               scene_graph, \
               self.trees[idx]
