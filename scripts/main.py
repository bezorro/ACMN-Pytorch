from __future__ import print_function, division
import torch
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import argparse
import torch.optim as optim
from shutil import copyfile
from tensorboardX import SummaryWriter
import time
import sys
from tqdm import tqdm

import model.tree_module as M
from data.data import getDataloader
from myutils import *

########################################################
## 					  Set Parameters			 	  ##
########################################################
parser = argparse.ArgumentParser(description='PyTorch TREE_BPMEM(Openended)')
# training settings
parser.add_argument('--batch_size', type=int, default=2, help="training batch size")
parser.add_argument('--max_epoch', type=int, default=1000, help='max epoches')
parser.add_argument('--lr', type=float, default=5e-6, help='learning Rate. Default=0.0002')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay. Default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='adam Beta1. Default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='adam Beta2. Default=0.999')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=123')
parser.add_argument('--display_interval', type=int, default=100, help='display loss after each several iters')
parser.add_argument('--display_att', type=bool, default=False, help='whether display attention map or not')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu?')
parser.add_argument('--load_lookup', type=bool, default=False, help='dir to tensorboard logs')
parser.add_argument('--train_set', type=str, default='clevr', help='training datatset')
# model settings
parser.add_argument('--encode', type=str, default='LSTM', help='how to encode sentences')
parser.add_argument('--sentence_emb', type=int, default=2048, help='embedding size of sentences')
parser.add_argument('--word_emb_size', type=int, default=300, help='embedding size of words')
parser.add_argument('--img_emb', type=int, default=128)
parser.add_argument('--commom_emb', type=int, default=256, help='commom embedding size of sentence and image')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout ration. Default=0.0')

# file settings
parser.add_argument('--vocab_dir', type=str, default='../data/clevr/clevr_qa_dir/', help='dir of qa pairs')
parser.add_argument('--qa_dir', type=str, default='../data/clevr/clevr_qa_dir/parsed_tree/', help='dir of qa pairs')
parser.add_argument('--clevr_img_h5', type=str, default='../data/clevr/clevr_res101/', help='dir of vqa image feature')
# parser.add_argument('--resume', type=str, default=None, help='resume file name')
parser.add_argument('--resume', type=str, default='../logs/clevr_pretrained_model.pth', help='resume file name')
parser.add_argument('--logdir', type=str, default='../logs/test', help='dir to tensorboard logs')
opt = parser.parse_args()
dataloader_train, dataloader_test = getDataloader(opt)
opt.default = 'kaiminginit_modClassifier_res101_noShare_biLSTMword_AdvFCshift_d.0'
print('dataset loaded')
print(opt)
########################################################
## 			         Basic Settings			   	  	  ##
########################################################
if not os.path.isdir(opt.logdir): os.mkdir(opt.logdir)
if opt.logdir is not None:
	f = open(opt.logdir + '/params.txt','w')
	print(opt, file = f)
	f.close()
	copyfile('main.py', opt.logdir + '/main.py')
	copyfile('model/ResAttUnit.py', opt.logdir + '/ResAttUnit.py')
	copyfile('model/tree_module.py', opt.logdir + '/tree_module.py')
	copyfile('model/tree_abstract_module.py', opt.logdir + '/tree_abstract_module.py')
if opt.logdir is not None:
	writer = SummaryWriter(opt.logdir)
if opt.display_att == True:
	mvizer = vizer(opt, writer)
# torch.manual_seed(opt.seed)
########################################################
## 			   Build Model & Load Data			   	  ##
########################################################
print('Building network...')
criterion = nn.CrossEntropyLoss()
opt.total_batch_size = opt.batch_size
dataParallel = False
if dataParallel: opt.batch_size = opt.batch_size // torch.cuda.device_count()  # to build dataParallel Model
# net = M.tree_attention_DualPath(opt)
net = M.tree_attention_Residual(opt)
def init_weights(m):
	if isinstance(m, (nn.Conv2d, nn.Linear)):
		init.kaiming_uniform(m.weight)
net.apply(init_weights)

if opt.load_lookup == True: print('loaded lookup tables.')
optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay = opt.weight_decay)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
opt.batch_size = opt.total_batch_size
finished_epoch = 0

if opt.resume is not None:
	print('resume from ' + opt.resume)
	checkpoint = torch.load(opt.resume)
	net.load_state_dict(checkpoint['net'])
	# optimizer.load_state_dict(checkpoint['optimizer'])
	finished_epoch = checkpoint['epoch']
if opt.gpu == True:
	if dataParallel: net = model.treeNN_DataParallel(net).cuda()
	else: net.cuda()
	criterion = criterion.cuda()
print(net)
########################################################
## 					  Train Model			  		  ##
########################################################

def train(epoch):
	print('Trainning...')
	net.train()

	i_batch = 0
	num_iter = 0
	for sample_batched in tqdm(dataloader_train):
		inputs, labels, others = sample_batched

		# # wrap to Variable
		inputs = list(Variable(tensor) for tensor in inputs)
		labels = Variable(labels)

		# # to gpu
		if opt.gpu:
			inputs = list(var.cuda() for var in inputs)
			labels = labels.cuda()

		# forward and backward
		optimizer.zero_grad()
		predicts, node_values = net(*(inputs + [others[3]]))

		loss = criterion(predicts, labels)
		loss.backward()
		optimizer.step()

		# display
		if i_batch % opt.display_interval == 0:
			if opt.logdir is not None:
				writer.add_scalar('loss',loss.data[0],i_batch + len(dataloader_train) * (epoch - 1))

				if opt.display_att == True:
					mvizer.show_node_values(node_values[1], others[3], others[1], inputs[0].cpu(), labels, predicts)
		i_batch += 1
		if i_batch == 25000:
			checkpoint(epoch-0.5)
			test(epoch)

def test(epoch):
	if opt.train_set == 'vqa': return
	vdict_rev = read_json(os.path.join(opt.vocab_dir,'VocabRev.json'))
	adict_rev = read_json(os.path.join(opt.vocab_dir,'AnsVocabRev.json'))

	print('Evaluating...')

	net.eval()

	corrects = 0
	b = 0
	loss = 0

	for sample_batched in tqdm(dataloader_test):
		b = b+1
		inputs, labels, others = sample_batched

		# wrap to Variable
		inputs = tuple(Variable(tensor) for tensor in inputs)
		labels = Variable(labels)

		# to gpu
		if opt.gpu:
			inputs = list(var.cuda() for var in inputs)
			labels = labels.cuda()

	 	# do forward
		predicts, node_values = net(*(inputs + [others[3]]))
		loss += criterion(predicts, labels).data[0]
		corrects += predicts.max(1)[1].eq(labels).cpu().sum().data[0]

	if opt.display_att == True:
		mvizer.show_node_values(node_values[1], others[3], others[1], inputs[0].cpu(), labels, predicts)

	accuracy = 100. * corrects / len(dataloader_test.dataset)
	loss = loss / b
	print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(corrects, len(dataloader_test.dataset), accuracy))

	if opt.logdir is not None:
		writer.add_scalar('accuracy',accuracy, epoch)
		writer.add_scalar('test_loss',loss, epoch)

def checkpoint(epoch):
    model_out_path = opt.logdir + '/model_epoch_{}.pth'.format(epoch)
    checkpoint = {'net':net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    torch.save(checkpoint, model_out_path)
    print('Checkpoint saved to {}'.format(model_out_path))

for epoch in range(finished_epoch + 1, finished_epoch + opt.max_epoch + 1):
	train(epoch)
	checkpoint(epoch)
	test(epoch)
