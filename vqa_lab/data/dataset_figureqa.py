import h5py
import os.path
import numpy as np
import pandas as pd
import torch
import torch.nn.init
from torch.utils.data import Dataset
from PIL import Image
from vqa_lab.utils import read_json
from torchvision import transforms

class figureqaDataset(Dataset):
    """figureqa dataset."""
    def __init__(self, opt, mode = 'train'):
        """
        Args:
            qa_dir (string): Path to the h5 file with annotations.
            img_png (string): Path to the h5 file with image features
            mode (string): Mode of train or test
        """
        self.img_png    = opt.img_png
        self.qa_dir     = opt.qa_dir
        self.load_trees = opt.load_trees
        if mode   == 'train' : self.imgFolder = self.img_png + '/train1/png/'
        elif mode == 'val1'  : self.imgFolder = self.img_png + '/validation1/png/'
        elif mode == 'val2'  : self.imgFolder = self.img_png + '/validation2/png/'
        elif mode == 'test1' : self.imgFolder = self.img_png + '/no_annot_test1/png/'
        elif mode == 'test2' : self.imgFolder = self.img_png + '/no_annot_test2/png/'
        
        # qa h5
        if mode == 'train':
            file = h5py.File(os.path.join(self.qa_dir, 'annotation_figureqa_train.h5'), 'r')
            self.qas = {}
            self.qas['question'] = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id'] = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers'] = file['/answers'][:]
            file.close()
            if opt.load_trees : self.trees = read_json(os.path.join(self.img_png, 'parsed_tree/figureqa_train_sorted_trees.json'))
            # self.img_file = h5py.File(self.img_h5_folder+'/features_train.h5', 'r')
        elif 'val' in mode :
            file = h5py.File(os.path.join(self.qa_dir, 'annotation_figureqa_'+mode+'.h5'), 'r')
            self.qas = {}
            self.qas['question'] = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id'] = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers'] = file['/answers'][:]
            file.close()
            if opt.load_trees : self.trees = read_json(os.path.join(self.img_png, 'parsed_tree/figureqa_'+mode+'_sorted_trees.json'))
            # self.img_file = h5py.File(self.img_h5_folder+'/features_'+mode+'.h5', 'r')
        else:
            file = h5py.File(os.path.join(self.qa_dir, 'annotation_figureqa_'+mode+'.h5'), 'r')
            self.qas = {}
            self.qas['question'] = torch.from_numpy(np.int64(file['/ques_test'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_test'][:]))
            self.qas['img_id'] = torch.from_numpy(np.int32(file['/img_id_test'][:]))
            file.close()
            if opt.load_trees : self.trees = read_json(os.path.join(self.img_png, 'parsed_tree/figureqa_'+mode+'_sorted_trees.json'))
            # self.img_file = h5py.File(self.img_h5_folder+'/features_'+mode+'.h5', 'r')

        self.vocab     = read_json(os.path.join(self.qa_dir, 'Vocab.json'))
        self.vocab_inv = read_json(os.path.join(self.qa_dir, 'VocabRev.json'))
        self.ansVocab  = read_json(os.path.join(self.qa_dir, 'AnsVocab.json'))

        self.opt = {
                        'vocab_size'     : len(self.vocab)   , \
                        'out_vocab_size' : len(self.ansVocab), \
                        'sent_len'       : self.qas['question'].size(1)
                   }

        self.mode = mode

        self.preprocess = transforms.Compose([
           transforms.Resize((256,256)),
           transforms.ToTensor()
        ])

        print('    * figureqa-%s loaded' % mode)

    def __len__(self):

        return self.qas['question'].size(0)
    
    def __getitem__(self, idx):

        img_id = self.qas['img_id'][idx]

        if 'test' in self.mode: 

            answer = None
        
        else:

            answer = self.qas['answers'][idx][0] - 1
            answer = answer.item()

        qid = self.qas['question_id'][idx]
        
        def id2imgName(img_id, qid):

            return {
                     'train'  : self.imgFolder+'%d.png' % img_id , \
                     'val1'   : self.imgFolder+'%d.png' % img_id , \
                     'test'   : self.imgFolder+'%d.png' % img_id
                    }[self.mode]

        def load_image(img_name): ## load raw image
            img_tensor = self.preprocess(Image.open(img_name).convert('RGB'))
            return img_tensor

        img_name = id2imgName(img_id, qid)

        sample = { 
                    'question'   : self.qas['question'][idx]                    ,
                    'qid'        : qid                                          ,
                    'image'      : load_image(img_name)                         ,
                    'img_name'   : id2imgName(img_id, qid)                      ,
                    'answer'     : answer                                       ,
                }

        if self.load_trees :
            sample['tree'] = self.trees[idx]
        
        return sample
