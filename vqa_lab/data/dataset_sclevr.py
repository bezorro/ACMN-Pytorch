import h5py
import json
import os.path
import numpy as np
import torch
import torch.nn.init
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class sclevrDataset(Dataset):
    """clevr dataset."""
    def __init__(self, opt, mode = 'train'):
        """
        Args:
            qa_dir (string): Path to the h5 file with annotations.
            img_dir (string): Path to the h5 file with image features
            mode (string): Mode of train or test
        """

        self.qa_dir    = opt.qa_dir
        self.imgFolder = self.qa_dir + '/sclevr/images/'

        # qa h5
        if mode == 'train':

            file = h5py.File(os.path.join(self.qa_dir, 'annotation_sclevr_train.h5'), 'r')
            self.qas                = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_train'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_train'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_train'][:]))
            self.qas['answers']     = file['/answers'][:]
            file.close()
            self.types              = json.load(open(os.path.join(self.qa_dir, 'sclevr_train_type.json'), 'r'))
            self.img_info           = json.load(open(os.path.join(self.qa_dir, 'sclevr_train_info.json'), 'r'))
        
        else:

            file = h5py.File(os.path.join(self.qa_dir, 'annotation_sclevr_test.h5'), 'r')
            self.qas = {}
            self.qas['question']    = torch.from_numpy(np.int64(file['/ques_test'][:]))
            self.qas['question_id'] = torch.from_numpy(np.int64(file['/question_id_test'][:]))
            self.qas['img_id']      = torch.from_numpy(np.int32(file['/img_id_test'][:]))
            self.qas['answers']     = file['/answers'][:]
            file.close()

            self.types              = json.load(open(os.path.join(self.qa_dir, 'sclevr_test_type.json'), 'r'))
            self.img_info           = json.load(open(os.path.join(self.qa_dir, 'sclevr_test_info.json'), 'r'))

        # train_test json
        vocab = json.load(open(os.path.join(self.qa_dir, 'Vocab.json'), 'r'))
        ansVocab = json.load(open(os.path.join(self.qa_dir, 'AnsVocab.json'), 'r'))

        self.opt = {
                        'vocab_size'     : len(vocab)   , \
                        'out_vocab_size' : len(ansVocab), \
                        'sent_len'       : self.qas['question'].size(1)
                   }

        self.mode       = mode
        self.preprocess = transforms.Compose([
           transforms.Resize((128,128)),
           transforms.ToTensor()
        ])

        print('    * sclevr-%s loaded' % mode)

    def __len__(self):
        return self.qas['question'].size(0)
    
    def __getitem__(self, idx):

        img_id = self.qas['img_id'][idx]
        answer = self.qas['answers'][idx][0] - 1
        answer = answer.item()
        qid    = self.qas['question_id'][idx]
        
        def id2imgName(img_id, qid):

            if self.mode == 'train': return self.imgFolder+'/train/%d.png' % img_id
            else: return self.imgFolder + '/test/%d.png' % img_id

        def load_image(img_name):

            img_tensor = self.preprocess(Image.open(img_name).convert('RGB'))
            return img_tensor
        
        img_name = id2imgName(img_id, qid)
        img      = load_image(img_name)
        img_info = self.img_info[img_id]

        return { 
                    'question' : self.qas['question'][idx], \
                    'qid'      : qid                      , \
                    'answer'   : answer                   , \
                    'image'    : load_image(img_name)     , \
                    'img_name' : id2imgName(img_id, qid)
               }


