from myutils import *
import h5py
import os.path
import numpy as np
import pandas as pd
import torch
import torch.nn.init
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class clevrDataset(Dataset):
    """clevr dataset."""
    def __init__(self, opt, mode = 'train'):
        """
        Args:
            qa_dir (string): Path to the h5 file with annotations.
            img_dir (string): Path to the h5 file with image features
            mode (string): Mode of train or test
        """
        self.qa_dir = opt.qa_dir
        self.vocab_dir = opt.vocab_dir
        self.img_h5_folder = opt.clevr_img_h5
        self.imgFolder = '/home/caoqx/datasets/CLEVR_v1.0/images'

        # qa h5
        if mode == 'train':
            file = h5py.File(os.path.join(self.vocab_dir, 'annotation_clevr_train.h5'), 'r')
            self.qas = {}
            self.qas['question'] = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id'] = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers'] = file['/answers'][:]
            file.close()
            self.trees = read_json(os.path.join(self.qa_dir, 'clevr_train_sorted_remain_dep_trees.json'))
            self.img_file = h5py.File(self.img_h5_folder+'/features_train.h5', 'r')
            # self.scene = read_json(os.path.join(self.qa_dir, 'CLEVR_v1.0/scenes/CLEVR_train_scenes.json'))['scenes']
        elif mode == 'val':
            file = h5py.File(os.path.join(self.vocab_dir, 'annotation_clevr_val.h5'), 'r')
            self.qas = {}
            self.qas['question'] = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id'] = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers'] = file['/answers'][:]
            file.close()
            self.trees = read_json(os.path.join(self.qa_dir, 'clevr_val_sorted_remain_dep_trees.json'))
            self.img_file = h5py.File(self.img_h5_folder+'/features_val.h5', 'r')
            # self.scene = read_json(os.path.join(self.qa_dir, 'CLEVR_v1.0/scenes/CLEVR_val_scenes.json'))['scenes']
        else:
            file = h5py.File(os.path.join(self.vocab_dir, 'annotation_clevr_test.h5'), 'r')
            self.qas = {}
            self.qas['question'] = torch.from_numpy(np.int64(file['/ques_test'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_test'][:]))
            self.qas['img_id'] = torch.from_numpy(np.int32(file['/img_id_test'][:]))
            file.close()
            self.trees = read_json(os.path.join(self.qa_dir, 'clevr_test_sorted_remain_dep_trees.json'))
            self.img_file = h5py.File(self.img_h5_folder+'/features_test.h5', 'r')
        # train_test json
        vocab = read_json(os.path.join(self.vocab_dir, 'Vocab.json'))
        ansVocab = read_json(os.path.join(self.vocab_dir, 'AnsVocab.json'))
        opt.vocab_size = len(vocab)
        opt.out_vocab_size = len(ansVocab)

        opt.sent_len = self.qas['question'].size(1)
        self.mode = mode

        print('    * clevr-%s loaded' % mode)

    def __len__(self):
        return self.qas['question'].size(0)
    
    def __getitem__(self, idx):
        img_id = self.qas['img_id'][idx]
        if self.mode == 'test': answer = None
        else:
            answer = self.qas['answers'][idx][0] - 1
            answer = answer.item()
        qid = self.qas['question_id'][idx]
        
        def id2imgName(img_id, qid):
            if self.mode == 'train': return self.imgFolder+'/train/CLEVR_train_%06d.png' % img_id
            elif self.mode == 'val': return self.imgFolder+'/val/CLEVR_val_%06d.png' % img_id
            else: return self.imgFolder+'/test/CLEVR_test_%06d.png' % img_id

        def load_image(img_name):
            img_name = os.path.basename(img_name)
            return torch.from_numpy(np.array(self.img_file[img_name]))
        
        img_name = id2imgName(img_id, qid)
        scene_graph = []
        # if self.mode == 'test': scene_graph = []
        # else: scene_graph = self.scene[img_id]

        return self.qas['question'][idx], \
               qid, \
               answer, \
               load_image(img_name), \
               img_name, \
               scene_graph, \
               self.trees[idx]
