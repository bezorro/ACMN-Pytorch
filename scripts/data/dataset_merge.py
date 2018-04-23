from myutils import *
import h5py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class mergeDataset(Dataset):
    """vqa dataset."""
    def __init__(self, ds_list):
      self.ds_list = ds_list

      sum_len = 0
      self.sum_len0 = []
      self.sum_len1 = []
      for ds in ds_list:
        self.sum_len0.append(sum_len)
        sum_len += len(ds)
        self.sum_len1.append(sum_len)
      		
    def __len__(self):
        return sum( [len(ds) for ds in self.ds_list] )
    
    def __getitem__(self, idx):
    	for i in range(len(self.ds_list)):
    		if idx < self.sum_len1[i]:
    			return self.ds_list[i][idx - self.sum_len0[i]]
        