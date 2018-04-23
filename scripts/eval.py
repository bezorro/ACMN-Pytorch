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
from tensorboard import SummaryWriter
import time
from tqdm import tqdm

import model.tree_module as M
from data.data import getDataloader
from myutils import *

########################################################
## 					  Set Parameters			 	  ##
########################################################
parser = argparse.ArgumentParser(description='PyTorch TREE_BPMEM(Openended)')
# training settings
parser.add_argument('--batch_size', type=int, default=1, help="training batch size")
parser.add_argument('--max_epoch', type=int, default=1000, help='max epoches')
parser.add_argument('--lr', type=float, default=1e-4, help='learning Rate. Default=0.0002')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay. Default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9, help='adam Beta1. Default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='adam Beta2. Default=0.999')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--display_interval', type=int, default=100, help='display loss after each several iters')
parser.add_argument('--display_att', type=bool, default=False, help='whether display attention map or not')
parser.add_argument('--gpu', type=bool, default=True, help='use gpu?')
parser.add_argument('--load_lookup', type=bool, default=False, help='dir to tensorboard logs')
parser.add_argument('--train_set', type=str, default='clevr', help='training datatset')
# model settings
parser.add_argument('--encode', type=str, default='LSTM', help='how to encode sentences')
parser.add_argument('--bilinear', type=str, default='MLB', help='MUTAN | MLB')
parser.add_argument('--sentence_emb', type=int, default=2048, help='embedding size of sentences')
parser.add_argument('--word_emb_size', type=int, default=300, help='embedding size of words')
parser.add_argument('--img_emb', type=int, default=128, help='embedding size of images')
parser.add_argument('--pre_emb', type=int, default=310, help='embedding size of preprocess')
parser.add_argument('--mutan_r', type=int, default=5, help='R of MUTAN bilinear')
# parser.add_argument('--vocab_size', type=int, default=15534, help='vocabulary size of question')
# parser.add_argument('--out_vocab_size', type=int, default=3000, help='vocabulary size of question')
# parser.add_argument('--vocab_size', type=int, default=16926, help='vocabulary size of question')
parser.add_argument('--commom_emb', type=int, default=256, help='commom embedding size of sentence and image')
parser.add_argument('--num_output', type=int, default=4, help='number of choices')
# parser.add_argument('--sent_len', type=int, default=35, help='length of sentences')
parser.add_argument('--maxT', type=int, default=1, help='max time step')
parser.add_argument('--proposal', type=bool, default=False, help='use proposals')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout ration. Default=0.0')
parser.add_argument('--att_reward', type=float, default=0.3, help='attention reward. Default=0.3')
parser.add_argument('--kernal_size', type=int, default=3, help='size of conv padding')

# file settings
parser.add_argument('--vocab_dir', type=str, default='/home/caoqx/vqa-graph/clevr', help='dir of qa pairs')
parser.add_argument('--qa_dir', type=str, default='/home/caoqx/datasets/parsed_tree', help='dir of qa pairs')
parser.add_argument('--clevr_img_h5', type=str, default='/home/caoqx/vqa-graph/data/clevr_res101/', help='dir of vqa image feature')
# parser.add_argument('--vqa_img_h5', type=str, default='/home/caoqx/vqa-graph/data/vqa/features.h5', help='dir of vqa image feature')
# parser.add_argument('--resume', type=str, default=None, help='resume file name')
parser.add_argument('--resume', type=str, default='logs/clevr_kaiminginit_modClassifier_ResCNN_DPResOnly1BiLinear_res101_noShare_biLSTMword_AdvFCshift_d.0_resumeE6/model_epoch_13.pth', help='resume file name')
parser.add_argument('--logdir', type=str, default='logs/test', help='dir to tensorboard logs')
opt = parser.parse_args()
_, dataloader_test = getDataloader(opt)
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
torch.manual_seed(opt.seed)
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
	optimizer.load_state_dict(checkpoint['optimizer'])
	finished_epoch = checkpoint['epoch']
if opt.gpu == True:
	if dataParallel: net = model.treeNN_DataParallel(net).cuda()
	else: net.cuda()
	criterion = criterion.cuda()
print(net)
########################################################
## 					  Train Model			  		  ##
########################################################
def test(epoch):
	if opt.train_set == 'vqa': return
	vdict_rev = read_json(os.path.join(opt.vocab_dir,'VocabRev.json'))
	adict_rev = read_json(os.path.join(opt.vocab_dir,'AnsVocabRev.json'))

	print('Evaluating...')

	net.eval()

	corrects = 0
	loss = 0

	for sample_batched in tqdm(dataloader_test):
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
		# loss += criterion(predicts, labels).data[0]
		corrects += predicts.max(1)[1].eq(labels).cpu().sum().data[0]

	# scheduler.step(test_loss)
		if opt.display_att == True:
			mvizer.show_node_values(node_values[1], others[3], others[1], inputs[0].cpu(), labels, predicts)
			input()
		# for i in range(corrects.size(0)):
		# 	if corrects.data[i] == 0:
		# 		ques = ''
		# 		for wordInput in inputs[0][i]:
		# 			word = wordInput.cpu().data[0]
		# 			if word>0: ques+=vdict_rev[word-1] + ' '
		# 		print(ques,adict_rev[labels[i].data[0]],adict_rev[predicts.max(1)[1][i].data[0]])
	accuracy = 100. * corrects / len(dataloader_test.dataset)
	print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(corrects, len(dataloader_test.dataset), accuracy))

	# if opt.logdir is not None:
		# writer.add_scalar('accuracy',accuracy, epoch)
		# writer.add_scalar('test_loss',loss, epoch)

test(finished_epoch + 1)
