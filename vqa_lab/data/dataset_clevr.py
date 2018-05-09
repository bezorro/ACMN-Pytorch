import h5py
import json
import os.path
import numpy as np
import torch
import torch.nn.init
from torch.utils.data import Dataset
from vqa_lab.utils import read_json
from torchvision import transforms
from PIL import Image

class clevrDataset(Dataset):
    """clevr dataset."""
    def __init__(self, opt, mode = 'train'):
        """
        Args:
            opt.qa_dir(string)       : Path to the h5 file with annotations.
            opt.img_h5(string)       : Path to the h5 file with image features.
            opt.img_png(string)      : Path to the png file with images.
            opt.image_source(string) : Type of image source.
            opt.load_trees(bool)     : Load trees or not.
            opt.load_png(bool)       : Load png or not.
            mode(string)             : 
        """
        self.qa_dir        = opt.qa_dir
        self.img_h5        = opt.img_h5
        self.img_png       = opt.img_png
        self.image_source  = opt.image_source
        self.load_trees    = opt.load_trees
        self.load_png      = opt.load_png
        self.mode          = mode

        # qa h5
        if mode == 'train':

            file                    = h5py.File(os.path.join(self.qa_dir, 'annotation_clevr_train.h5'), 'r')
            self.qas                = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers']     = file['/answers'][:]
            file.close()
            if self.load_trees:
                self.trees          = read_json(os.path.join(self.qa_dir, 'parsed_tree/clevr_train_sorted_remain_dep_trees.json'))
            
        elif mode == 'val':

            file                    = h5py.File(os.path.join(self.qa_dir, 'annotation_clevr_val.h5'), 'r')
            self.qas                = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers']     = file['/answers'][:]
            file.close()
            if self.load_trees:
                self.trees          = read_json(os.path.join(self.qa_dir, 'parsed_tree/clevr_val_sorted_remain_dep_trees.json'))
            
        else:

            file                    = h5py.File(os.path.join(self.qa_dir, 'annotation_clevr_test.h5'), 'r')
            self.qas                = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_test'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_test'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_test'][:]))
            file.close()
            if self.load_trees:
                self.trees          = read_json(os.path.join(self.qa_dir, 'parsed_tree/clevr_test_sorted_remain_dep_trees.json'))

        if self.image_source == 'h5' :
            self.img_file = h5py.File(os.path.join(self.img_h5, 'features_' + mode + '.h5'), 'r')
        elif self.image_source == 'png' :
            self.img_file = self.img_png

        # train_test json
        vocab            = read_json(os.path.join(self.qa_dir, 'Vocab.json'))
        ansVocab         = read_json(os.path.join(self.qa_dir, 'AnsVocab.json'))
        self.ansVocabRev = read_json(os.path.join(self.qa_dir, 'AnsVocabRev.json'))
        self.VocabRev    = read_json(os.path.join(self.qa_dir, 'VocabRev.json'))

        self.opt = {
                        'vocab_size'     : len(vocab)   , \
                        'out_vocab_size' : len(ansVocab), \
                        'sent_len'       : self.qas['question'].size(1)
                   }

        self.preprocess = transforms.Compose([
           transforms.Scale((224,224)),
           transforms.ToTensor()
        ])

        print('    * clevr-%s loaded' % mode)

    def __len__(self):

        return self.qas['question'].size(0)
    
    def __getitem__(self, idx):

        img_id = self.qas['img_id'][idx]

        qid = self.qas['question_id'][idx]
        
        def id2imgName(img_id, qid):

            return self.mode + '/CLEVR_' + self.mode + '_%06d.png' % img_id
        
        img_name = id2imgName(img_id, qid)

        def get_image_feature(iname):

            if self.image_source == 'h5' :
                return torch.from_numpy(np.array(self.img_file[os.path.basename(iname)]))
            else : return self.preprocess(Image.open(os.path.join(self.img_file, iname)).convert('RGB'))

        sample = { 
                    'question' : self.qas['question'][idx]                , \
                    'qid'      : qid                                      , \
                    'image'    : get_image_feature(img_name)              , \
                    'img_name' : id2imgName(img_id, qid)                  , \
               }

        if self.mode == 'train' or self.mode == 'val' : 
            sample['answer'] = (self.qas['answers'][idx][0] - 1).item()

        if self.load_trees :
            sample['tree'] = self.trees[idx]

        if self.load_png :
            sample['img_png'] = transforms.ToTensor()(Image.open(os.path.join(self.img_png, id2imgName(img_id, qid))).convert('RGB'))

        return sample

