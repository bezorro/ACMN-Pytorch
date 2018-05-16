from __future__ import division
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import torch
import torch.nn.functional as F
import json
import os
import torchvision.models as models

def read_json(fname):
    file = open(fname, 'r')
    res = json.load(file)
    file.close()
    return res

class ResBlock(nn.Module):
    def __init__(self):
        super(ResBlock, self).__init__()
        # nets
        self.conv1 = nn.Conv2d(130, 128, 1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, v, coord_tensor):
        v = torch.cat( [v, coord_tensor.expand(v.size(0), 2, v.size(2), v.size(3))] , 1) # (B, 130, 14, 14)
        v1 = F.relu(self.conv1(v)) # (B, 128, 14, 14)

        v = self.conv2(v1)
        v = self.bn(v)
        v = F.relu(v)

        v = v + v1
        return v

class CNN(nn.Module):
    """docstring for tree"""
    def __init__(self):
        super(CNN, self).__init__()
        self.featMapH     = 14
        self.featMapW     = 14
        self.num_proposal = self.featMapH*self.featMapW
        # nets
        self.conv         = nn.Conv2d(1026, 128, 3, padding = 1)
        self.bn           = nn.BatchNorm2d(128)
        self.res1         = ResBlock()
        self.res2         = ResBlock()

        def cvt_coord(i):
            return [(i / self.featMapW - (self.featMapH // 2)) / (self.featMapH / 2.), (i % self.featMapW - (self.featMapW // 2)) / (self.featMapW / 2.)]
        self.coord_tensor = torch.FloatTensor(self.num_proposal, 2).cuda()
        np_coord_tensor   = np.zeros((self.num_proposal, 2))
        for i in range(self.num_proposal):
            np_coord_tensor[i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.copy_(torch.from_numpy(np_coord_tensor))
        self.coord_tensor = self.coord_tensor.view(1,self.featMapH,self.featMapW,2).transpose_(2,3).transpose_(1,2)
        self.coord_tensor = Variable(self.coord_tensor)

    def forward(self, img):
        img = torch.cat([img, self.coord_tensor.expand(img.size(0), 2, img.size(2), img.size(3))] , 1)
        img = F.relu(self.bn(self.conv(img)))

        img = self.res1(img, self.coord_tensor)
        img = self.res2(img, self.coord_tensor)

        return img

class tree_attention_abstract_DP(nn.Module):
    """docstring for tree"""
    def __init__(self, opt):
        super(tree_attention_abstract_DP, self).__init__()
        self.featMapH       = 14
        self.featMapW       = 14
        self.num_proposal   = self.featMapH * self.featMapW
        self.input_emb_size = opt.sentence_emb
        self.dropout        = opt.dropout
        self.img_emb_size   = opt.img_emb
        self.gpu            = opt.gpu
        self.sent_len       = opt.sent_len
        self.lookup         = nn.Embedding(opt.vocab_size + 1, opt.word_emb, padding_idx = 0) #mask zero
        self.q_LSTM         = nn.LSTM(300, 1024, 1, bidirectional=True)
        self.CNN            = CNN()

    def forward(self, que, img, tree):
        # img(B, 3, self.featMapHW, self.featMapHW)
        que_enc, que_enc_sent = self._encoder(que) # (batch_size, emb)
        batch_size = que_enc.size(0)
        #-------img prepro------------------
        def l2normalizer(img):
            img = img.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 1024)
            img = F.normalize(img)
            img = img.view(-1, self.featMapH, self.featMapW, 1024).transpose(2, 3).transpose(1, 2).contiguous()
            return img
        img = l2normalizer(img)
        img = self.CNN(img) # (B, 128, H, W)
        
        #---------end-----------------------
        self.bmax_height = max((t[-1]['height'] for t in tree)) + 1

        #--------get selected map---------------
        # return a (B, H ,[x,x,x]) list
        def get_selected_map(tree, maxh):
            selected_map = []
            que_len = []
            for i in range(batch_size):
                length = 0
                tmap = [[] for _ in range(maxh)]
                for j in range(len(tree[i])):
                    h = tree[i][j]['height']
                    tmap[h].append(j)
                    length+=len(tree[i][j]['word'])
                selected_map.append(tmap)
                que_len.append(length)
            return selected_map, que_len
        #--------------end----------------------
        node_values = [[len(tree[i]) * [o] for i in range(batch_size)] for o in self.init_node_values()] # init node_values
        # node_values = [K, B, tree_len(varlen)] , a init value in each grid
        selected_map, que_len = get_selected_map(tree, self.bmax_height)
        # selected_map = [B, H, hnodes_pos(varlen)]

        def get_tw_tn(bs_sum, height):
            tmp_word_inputs = torch.LongTensor(bs_sum,self.sent_len).zero_()
            tmp_q_inputs = Variable(torch.FloatTensor(bs_sum,self.input_emb_size).zero_()).cuda()
            tree_nodes = []
            cnt = 0
            for i in range(batch_size):
                for j in range(len(selected_map[i][height])): 
                    t_node = tree[i][selected_map[i][height][j]]
                    word_idx = self.sent_len - que_len[i] + t_node['index']-1
                    assert word_idx>=0 and word_idx<self.sent_len, 'word_idx: {}, {}, {}, {}'.format(word_idx,self.sent_len,que_len[i],t_node['index'])
                    tmp_q_inputs[cnt] = que_enc_sent[word_idx][i]
                    tree_nodes.append(t_node)
                    cnt += 1

            return tmp_word_inputs, tree_nodes, ijlist, tmp_q_inputs

        def put_back_nv(tmp_outputs, node_values, height):
            for k in range(len(tmp_outputs)):
                cnt = 0
                for i in range(len(node_values[k])):
                    for j in selected_map[i][height]:
                        node_values[k][i][j] = tmp_outputs[k][cnt]
                        cnt += 1

        for height in range(self.bmax_height):
            bs_sum = sum([len(selected_map[i][height]) for i in range(batch_size)])
            ijlist = []
            for i in range(batch_size): 
                for j in selected_map[i][height]: ijlist.append((i, j))
            tmp_inputs, node_que_enc = self.load_tmp_values(img, que_enc, tree, ijlist, node_values, height) # [4, TS(batch_sum_mnum, maxson, f1, f2...) ]
            tmp_word_inputs, tree_nodes, ijlist, tmp_q_inputs = get_tw_tn(bs_sum, height)
            tmp_outputs = self.node[height](tmp_inputs, tmp_q_inputs, tree_nodes, node_que_enc)
            put_back_nv(tmp_outputs, node_values, height)

        predict = self.root_to_att(que_enc, img, tree, node_values)
        return predict, node_values

    def init_node_values(self):
        raise NotImplementedError

    def load_tmp_values(self, img, tree, ijlist, node_values, height):
        raise NotImplementedError

    def root_to_att(self, img, node_values):
        raise NotImplementedError

    def _encoder(self, sentence):
        # sentence vocab begins from 1.And 0 means no word
        # input size: -1 * sentence_length
        # output size: -1 * embedding_size
        emb = self.lookup(sentence.view(-1, self.sent_len))
        qenc, _ = self.q_LSTM(emb.transpose(0,1))
        qenc    = F.normalize(qenc.view(-1, self.input_emb_size)).view(self.sent_len, -1, self.input_emb_size)
        enc     = qenc[-1]
        return enc, qenc
