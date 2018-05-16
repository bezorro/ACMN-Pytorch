from torch.autograd import Variable
import torch.nn as nn
import torch
import torch.nn.functional as F
import json
import os
from .tree_abstract_module import tree_attention_abstract_DP
from . import ResAttUnit

class tree_attention_Residual(tree_attention_abstract_DP):
	def __init__(self, opt):
		super(tree_attention_Residual, self).__init__(opt)
		self.num_node_feature = 256
		self.max_height = 12
		self.dropout=opt.dropout
		self.attNum = 1

		self.node = []
		self.nodeAtt = None
		for i in range(self.max_height):
			self.node.append(ResAttUnit.NodeBlockResidual(i, self.img_emb_size, self.num_node_feature, self.input_emb_size, self.attNum, self.dropout))
			self.add_module('node_'+str(i), self.node[i])

		## AnsModule
		self.fc0 = nn.Linear(self.attNum * self.num_node_feature, 512)
		self.bn0 = nn.BatchNorm1d(512)
		self.fc1 = nn.Linear(512, 1024)
		self.bn1 = nn.BatchNorm1d(1024)
		self.fc2 = nn.Linear(1024, opt.out_vocab_size)

	def init_node_values(self):
		return (None, None)

	def load_tmp_values(self, img, que_enc, tree, tree_list, node_values, height):

		node_que_enc = Variable(torch.FloatTensor(len(tree_list), self.input_emb_size).zero_()).cuda()
		if height == 0:
			input2 = Variable(torch.FloatTensor(len(tree_list), self.img_emb_size, self.featMapH, self.featMapW).zero_()).cuda()
			for i, v in enumerate(tree_list):
				ibatch = v[0]
				input2[i] = img[ibatch]
				node_que_enc[i] = que_enc[ibatch]
			return (None, None, input2), node_que_enc
		else:
			maxlen = 6
			input0 = Variable(torch.FloatTensor(len(tree_list), maxlen, self.num_node_feature).zero_()).cuda()
			input1 = Variable(torch.FloatTensor(len(tree_list), maxlen, self.attNum, self.featMapH, self.featMapW).zero_()).cuda()
			input2 = Variable(torch.FloatTensor(len(tree_list), self.img_emb_size, self.featMapH, self.featMapW).zero_()).cuda()
			input3 = Variable(torch.FloatTensor(len(tree_list), maxlen, self.num_node_feature).zero_()).cuda()
			for i, v in enumerate(tree_list):
				ibatch = v[0]
				jpos = v[1]
				for son_cnt in range(len(tree[ibatch][jpos]['child'])):
					son_pos = tree[ibatch][jpos]['child'][son_cnt]
					son_height = tree[ibatch][son_pos]['height']
					input3[i, son_cnt] = node_values[0][ibatch][son_pos]
					if tree[ibatch][son_pos]['dep'] == 1: input0[i, son_cnt] = node_values[0][ibatch][son_pos] #clausal: trans convResFeat
					if tree[ibatch][son_pos]['dep'] > 10: input1[i, son_cnt] = node_values[1][ibatch][son_pos] #modifier: trans attMap
				input2[i] = img[ibatch]
				node_que_enc[i] = que_enc[ibatch]

		return (input0,input1,input2,input3), node_que_enc

	def answer(self, feat):

		x_ = self.fc0(feat)
		x_ = self.bn0(x_)
		x_ = F.relu(x_)
		x_ = self.fc1(x_)
		x_ = self.bn1(x_)
		x_ = F.relu(x_)
		x_ = self.fc2(F.dropout(x_, self.dropout, training = self.training))

		return x_

	def root_to_att(self, que_enc, img, tree, node_values):

		max_feat_len = self.attNum * self.num_node_feature
		batch_size = img.size(0)
		res = Variable(torch.FloatTensor(batch_size, max_feat_len).zero_()).cuda()
		for i in range(batch_size):
			j = len(tree[i]) - 1
			res[i] = node_values[0][i][j]

		predict = self.answer(res)
		return predict