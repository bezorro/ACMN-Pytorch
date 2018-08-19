import torch.nn as nn
import torch
import torch.nn.functional as F
import os
from .CondBatchNorm2d import CondBatchNorm2d

class EncoderNet(nn.Module):
	def __init__(self, emb_size, vocab_size, hidden_size, gpu):
		super(EncoderNet, self).__init__()

		# opts
		self.input_voc_size = vocab_size  # 81
		self.input_emb_size = emb_size # 200
		self.hidden_size = hidden_size # 4096
		self.gpu = gpu

		# nets
		self.lookup = nn.Embedding(self.input_voc_size + 1, self.input_emb_size, padding_idx = 0)
		self.gru = nn.GRU(self.input_emb_size, self.hidden_size, batch_first = True)

	def forward(self, question):
		qemb = self.lookup(question) # (B, slen, 200)
		_, qenc = self.gru(qemb)

		return qenc # (B, 4096)

class CNN(nn.Module):
	"""docstring for CNN"""
	def __init__(self, in_c, out_c):
		super(CNN, self).__init__()

		self.input_channel = in_c
		self.output_channel = out_c

		self.conv1 = nn.Conv2d(self.input_channel, self.output_channel, 3, stride = 2, padding = 1)
		self.bn1 = nn.BatchNorm2d(self.output_channel)
		self.conv2 = nn.Conv2d(self.output_channel, self.output_channel, 3, stride = 2, padding = 1)
		self.bn2 = nn.BatchNorm2d(self.output_channel)
		self.conv3 = nn.Conv2d(self.output_channel, self.output_channel, 3, stride = 2, padding = 1)
		self.bn3 = nn.BatchNorm2d(self.output_channel)
		self.conv4 = nn.Conv2d(self.output_channel, self.output_channel, 3, stride = 2, padding = 1)
		self.bn4 = nn.BatchNorm2d(self.output_channel)

	def forward(self, image):
		image_feature = self.bn1(F.relu(self.conv1(image)))
		image_feature = self.bn2(F.relu(self.conv2(image_feature)))
		image_feature = self.bn3(F.relu(self.conv3(image_feature)))
		image_feature = self.bn4(F.relu(self.conv4(image_feature)))
		
		return image_feature

class ResBlock(nn.Module):
	"""docstring for ResBlock"""
	def __init__(self, dropout, in_h, in_w, cuda):
		super(ResBlock, self).__init__()

		self.dropout = dropout
		# nets
		self.conv1 = nn.Conv2d(128, 128, 1)
		self.conv2 = nn.Conv2d(128, 128, 3, padding = 1)
		self.bn = nn.BatchNorm2d(128)
		self.cbn1 = CondBatchNorm2d(128, 4096)

	def forward(self, q, v):

		v1 = F.relu(self.conv1(v)) # (B, 128, 14, 14)

		v = self.conv2(v1)
		v = self.bn(v)
		v = self.cbn1(v, q)
		v = F.relu(v)

		v = v + v1

		return v # (B, 128, 14, 14)

class Classifier(nn.Module):
	"""docstring for Classifier"""
	def __init__(self, out_s, dropout, cuda):
		super(Classifier, self).__init__()

		self.dropout = dropout
		self.conv = nn.Conv2d(128, 512, 1, padding = 0)
		self.bn1 = nn.BatchNorm2d(512, affine=True)
		self.fc1 = nn.Linear(512, 1024)
		self.bn2 = nn.BatchNorm1d(1024, affine=True)
		self.fc2 = nn.Linear(1024, out_s)

	def forward(self, v):

		v = F.relu(self.bn1(self.conv(v))) # (B, 512, 14, 14)
		v = F.max_pool2d(v, v.size(2)).view(-1, 512) # (B, 512) concate 4096 after max pooling
		v = F.relu(self.bn2(self.fc1(v)))
		v = self.fc2(v)

		return v

class ImgPrePro(nn.Module):
	"""docstring for ImgPrePro"""
	def __init__(self, cuda):
		super(ImgPrePro, self).__init__()

		self.conv1 = nn.Conv2d(1024, 512, 3, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(512)
		self.conv2 = nn.Conv2d(512, 256, 3, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(256)
		self.conv3 = nn.Conv2d(256, 256, 3, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(128)

		self.conv = nn.Conv2d(128, 128, 3, padding = 1)
		self.bn = nn.BatchNorm2d(128, affine=True)

	def forward(self, v):

		v = self.conv1(v)
		v = F.relu(v)
		v = self.bn1(v)
		v = self.conv2(v)
		v = F.relu(v)
		v = self.bn2(v)
		v = self.conv3(v)
		v = F.relu(v)
		v = self.bn3(v)
		v = self.conv4(v)
		v = F.relu(v)
		v = self.bn4(v)# (B, 128, 16, 16)

		return F.relu(self.bn(self.conv(v))) # (B, 26, 16, 16)

class ResBlockNet(nn.Module):

	def __init__(self, opt):
		super(ResBlockNet, self).__init__()

		# self.img_channel = opt.img_channel
		self.conv_channel = 128
		self.vocab_size = opt.vocab_size
		self.word_emb = opt.word_emb
		self.gru_hidden = 4096
		self.gpu = opt.gpu
		self.output_size = opt.out_size
		self.dropout = .0

		self.encode = EncoderNet(self.word_emb, self.vocab_size, self.gru_hidden, self.gpu)
		#self.cnn = CNN(self.img_channel, self.conv_channel)
		self.img_pre = ImgPrePro(self.gpu)
		self.resblock1 = ResBlock(self.dropout, 8, 8, self.gpu)
		self.resblock2 = ResBlock(self.dropout, 8, 8, self.gpu)
		self.resblock3 = ResBlock(self.dropout, 8, 8, self.gpu)
		self.resblock4 = ResBlock(self.dropout, 8, 8, self.gpu)
		self.classifier = Classifier(self.output_size, self.dropout, self.gpu)

	def forward(self, question, image):

		qenc = self.encode(question) # (B, 4096)
		img_feat = self.img_pre(image) # (B, 128, 8, 8) 

		img_feat = self.resblock1(qenc, img_feat) # (B, 128, 16, 16) 
		img_feat = self.resblock2(qenc, img_feat) # (B, 128, 16, 16) 
		img_feat = self.resblock3(qenc, img_feat) # (B, 128, 16, 16)
		img_feat = self.resblock4(qenc, img_feat) # (B, 128, 16, 16) 

		res = self.classifier(img_feat)

		return res