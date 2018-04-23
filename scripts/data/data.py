import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from .dataset_vqa import vqaDataset
from .dataset_clevr import clevrDataset
from .dataset_sclevr import sclevrDataset
from .dataset_merge import mergeDataset

def my_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    question, qid, answer, image, image_name, scene_graph, tree = zip(*batch)
    
    image = torch.stack(image)
    question = torch.stack(question)
    qid = torch.LongTensor(qid)
    answer = torch.LongTensor(answer)

    inputs = question, image
    labels = answer
    others = qid, image_name, scene_graph, tree
        
    return inputs, labels, others

def getDataloader(opt):
    ds_list = []
    print('Loading datasets...')
    
    if opt.train_set == 'vqa':
        dataset_train = vqaDataset(opt, 'train')
        dataset_test = dataset_train

    if opt.train_set == 'sclevr':
        dataset_train = sclevrDataset(opt, 'train')
        dataset_test = sclevrDataset(opt, 'test')
        opt.threads = 4

    if opt.train_set == 'clevr':
        # dataset_train = mergeDataset([clevrDataset(opt, 'train'), clevrDataset(opt, 'val')])
        # dataset_test = clevrDataset(opt, 'test')
        dataset_train = clevrDataset(opt, 'train')
        dataset_test = clevrDataset(opt, 'val')

    print('datasets loaded')

    # return DataLoader(dataset_test, \
    #         batch_size = opt.batch_size, \
    #         collate_fn = my_collate, \
    #         num_workers = opt.threads, \
    #         shuffle= False, \
    #         drop_last = False)

    return DataLoader(dataset_train, \
                    batch_size = opt.batch_size, \
                    collate_fn = my_collate, \
                    num_workers = opt.threads, \
                    shuffle= True, \
                    drop_last = True), \
            DataLoader(dataset_test, \
            batch_size = opt.batch_size, \
            collate_fn = my_collate, \
            num_workers = opt.threads, \
            shuffle= False, \
            drop_last = False)