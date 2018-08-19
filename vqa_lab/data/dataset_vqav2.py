import h5py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.init
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from vqa_lab.utils import read_json

class vqa2Dataset(Dataset):
    """vqa dataset."""
    def __init__(self, opt, mode = 'train'):
        """
        Args:
            img_dir (string): Path to the h5 file with image features
        """
        self.qa_dir       = opt.qa_dir
        self.img_h5       = opt.img_h5
        self.img_BU       = opt.img_BU
        self.img_jpg      = opt.img_jpg
        self.image_source = opt.image_source
        self.load_trees   = opt.load_trees
        self.load_jpg     = opt.load_jpg

        # qa h5
        if mode == 'train' or mode ==  'val': #train or val

            # file = h5py.File(os.path.join(opt.qa_dir, 'annotation_' + mode + 'v2_noskip.h5'), 'r')
            file = h5py.File(os.path.join(opt.qa_dir, 'annotation_' + mode + 'v2.h5'), 'r')
            self.qas = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers']     = torch.from_numpy(np.int64(file['/answers'][:]))
            file.close()

        else:#test

            file = h5py.File(os.path.join(opt.qa_dir, 'annotation_' + mode + 'v2.h5'), 'r')
            self.qas = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_test'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_test'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_test'][:]))
            file.close()
            
        # img feature
        if self.image_source == 'h5' :
            if mode == 'test' or  mode == 'test-dev':
                self.img_file = h5py.File(os.path.join(self.img_h5, 'features_test.h5'), 'r')
            else :
                self.img_file = h5py.File(os.path.join(self.img_h5, 'features_train-val.h5'), 'r')
        elif self.image_source == 'BU' :
            if mode == 'test' or  mode == 'test-dev':
                self.img_file = h5py.File(os.path.join(self.img_BU, 'features_test.h5'), 'r')
            else :
                self.img_file = h5py.File(os.path.join(self.img_BU, 'features_train-val.h5'), 'r')
        elif self.image_source == 'jpg' :
            self.img_file = os.path.join(self.img_jpg, {'train':'train2014', 'val':'val2014','test':'test2015','test-dev':'test2015'}[mode])

        if self.load_trees:
            self.trees = read_json(os.path.join(opt.qa_dir, 'parsed_tree' , mode + 'v2_sorted_trees.json'))

        # train_test json
        self.cur_list = range(len(self.qas['question_id']))

        vocab    = read_json(os.path.join(self.qa_dir, 'Vocab.json'))
        ansVocab = read_json(os.path.join(self.qa_dir, 'AnsVocab.json'))

        self.opt = {
                        'vocab_size'     : len(vocab)   , \
                        'out_vocab_size' : len(ansVocab), \
                        'sent_len'       : self.qas['question'].size(1)
                   }

        self.mode = mode

        print('    * vqa2-%s loaded' % mode)

        self.preprocess = transforms.Compose([
           transforms.Resize((256,256)),
           transforms.ToTensor()
        ])

    def __len__(self):

        return len(self.cur_list)
    
    def __getitem__(self, idx):

        idx    = self.cur_list[idx]  
        img_id = self.qas['img_id'][idx]
        qid    = self.qas['question_id'][idx]
        
        def id2imgName(img_id, qid, jpg = False):

            if self.image_source == 'BU' and jpg == False :

                return {
                         'train'    : 'COCO_train-val2014_%012d.jpg' % img_id ,
                         'val'      : 'COCO_train-val2014_%012d.jpg' % img_id   ,
                         'test'     : 'COCO_test2015_%012d.jpg' % img_id  ,
                         'test-dev' : 'COCO_test2015_%012d.jpg' % img_id  ,
                        }[self.mode]
            else:

                return {
                         'train'    : 'COCO_train2014_%012d.jpg' % img_id ,
                         'val'      : 'COCO_val2014_%012d.jpg' % img_id   ,
                         'test'     : 'COCO_test2015_%012d.jpg' % img_id  ,
                         'test-dev' : 'COCO_test2015_%012d.jpg' % img_id  ,
                        }[self.mode]

        def get_answer():

            answer = self.qas['answers'][idx] - 1
            rn = torch.randperm((answer >= 0).sum())[0]
            return answer[rn]
        
        img_name = id2imgName(img_id, qid, jpg=False)

        def get_image_feature():

            if self.image_source == 'h5' or self.image_source == 'BU':
                return torch.from_numpy(np.array(self.img_file[img_name]))
            else : return self.preprocess(Image.open(os.path.join(self.img_file, img_name)).convert('RGB'))

        sample = {
                    'question'   : self.qas['question'][idx]                    ,
                    'qid'        : qid                                          ,
                    'image'      : get_image_feature()                          ,
                    'img_name'   : id2imgName(img_id, qid, jpg=True)            ,
                }

        if self.mode == 'train' or self.mode == 'val' : 
            sample['answer'], sample['raw_answer'] = get_answer(), self.qas['answers'][idx] - 1

        if self.load_trees :
            sample['tree'] = self.trees[idx]

        if self.load_jpg :
            img_file = os.path.join(self.img_jpg, {'train':'train2014', 'val':'val2014','test':'test2015','test-dev':'test2015'}[self.mode])
            sample['img_jpg'] = self.preprocess(Image.open(os.path.join(img_file, id2imgName(img_id, qid, jpg=True))).convert('RGB'))

        return sample
