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

class sgkbDataset(Dataset):
    """sgkb dataset."""
    def __init__(self, opt, mode = 'train'):
        """
        Args:
            opt.qa_dir(string)       : Path to the h5 file with annotations.
            opt.img_h5(string)       : Path to the h5 file with image features.
            opt.img_jpg(string)      : Path to the jpg file with images.
            opt.image_source(string) : Type of image source.
            opt.load_jpg(bool)       : Load jpg or not.
            mode(string)             : 
        """
        self.qa_dir        = opt.qa_dir
        self.img_h5        = opt.img_h5
        self.img_jpg       = opt.img_jpg
        self.img_npy       = opt.img_npy
        self.image_source  = opt.image_source
        self.load_jpg      = opt.load_jpg
        self.mode          = mode

        # qa h5
        if mode == 'train':

            file                    = h5py.File(os.path.join(self.qa_dir, 'annotation_sgkb_train.h5'), 'r')
            self.qas                = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers']     = file['/answers'][:]
            file.close()
            
        elif mode == 'val':

            file                    = h5py.File(os.path.join(self.qa_dir, 'annotation_sgkb_val.h5'), 'r')
            self.qas                = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers']     = file['/answers'][:]
            file.close()
            
        else:

            file                    = h5py.File(os.path.join(self.qa_dir, 'annotation_sgkb_test.h5'), 'r')
            self.qas                = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_test'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_test'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_test'][:]))
            file.close()

        if self.image_source == 'h5' :
            self.img_file = h5py.File(os.path.join(self.img_h5, 'features_' + mode + '.h5'), 'r')
        elif self.image_source == 'jpg' :
            self.img_file = self.img_jpg
        elif self.image_source == 'npy' :
            self.img_file = self.img_npy

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
           transforms.Resize((224,224)),
           transforms.ToTensor()
        ])

        print('    * sgkb-%s loaded' % mode)

    def __len__(self):

        return self.qas['question'].size(0)
    
    def __getitem__(self, idx):

        img_id = self.qas['img_id'][idx]

        qid = self.qas['question_id'][idx]
        
        def id2imgName(img_id):

            return str(int(img_id)) + '.jpg'
        
        img_name = id2imgName(img_id)

        def get_image_feature(iname):

            if self.image_source == 'h5' :
                return torch.from_numpy(np.array(self.img_file[os.path.basename(iname)]))
            elif self.image_source == 'jpg': 
                return self.preprocess(Image.open(os.path.join(self.img_file, iname)).convert('RGB'))
            elif self.image_source == 'npy':
                return torch.from_numpy(np.load(os.path.join(self.img_file, iname + '.npy'), encoding='latin1'))    

        sample = { 
                    'question' : self.qas['question'][idx]                , \
                    'qid'      : qid                                      , \
                    'image'    : get_image_feature(img_name)              , \
                    'img_name' : id2imgName(img_id)                       , \
               }

        if self.mode == 'train' or self.mode == 'val' : 
            sample['answer'] = (self.qas['answers'][idx][0]).item()

        if self.load_jpg :
            sample['img_jpg'] = transforms.ToTensor()(Image.open(os.path.join(self.img_jpg, id2imgName(img_id))).convert('RGB'))

        return sample

