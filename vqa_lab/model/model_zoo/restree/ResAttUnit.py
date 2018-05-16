import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

class Bilnear(nn.Module):

    def __init__(self, num_inputs, num_outputs, input_emb_size, dropout=0.3):
        super(Bilnear, self).__init__()
        self.num_features = 2048
        self.num_outputs  = num_outputs
        self.fcv          = nn.Linear(num_inputs, self.num_features)
        self.bnv          = nn.BatchNorm1d(self.num_features)
        self.fcq          = nn.Linear(input_emb_size, self.num_features)
        self.fco          = nn.Linear(self.num_features, num_outputs)
        self.dropout      = dropout

    def forward(self, input, word):
        ## FC q
        word_W = F.dropout(word, self.dropout, training = self.training)
        w = self.fcq(word_W)
        ## FC v
        v = F.dropout(input, self.dropout, training = self.training)
        v = self.fcv(v)
        o = v * w
        o = F.relu(o)
        o = self.fco(o)  #extra linear from film
        return F.relu(o)

class MLB(nn.Module):

    def __init__(self, num_inputs, num_outputs, input_emb_size, dropout=0.3):
        super(MLB, self).__init__()
        self.num_features = 2048
        self.num_outputs = num_outputs
        self.conv1 = nn.Conv2d(num_inputs, self.num_features, 1, padding=0)
        # self.cbn1 = CondBatchNorm2d(self.num_features, input_emb_size, dropout)
        self.conv2 = nn.Conv2d(self.num_features, num_outputs, 1)
        self.fcq_w = nn.Linear(input_emb_size, self.num_features)
        self.dropout = dropout

    def forward(self, input, word):
        ## FC q
        word_W = F.dropout(word, self.dropout, training = self.training)
        weight = F.tanh(self.fcq_w(word_W)).view(-1, self.num_features, 1, 1)
        ## FC v
        v = input
        v = F.tanh(self.conv1(v))
        v = v * weight.expand_as(v) 
        v = self.conv2(v)
        return F.softmax(v.view(-1,14 * 14), dim=1).view(-1, self.num_outputs, 14, 14)

class AttImgSonModule(nn.Module):
    def __init__(self, num_inputs, num_outputs, input_emb_size, dropout=0.3):
        super(AttImgSonModule, self).__init__()
        self.num_features = 2048
        self.num_outputs  = num_outputs
        self.num_inputs   = num_inputs
        self.conv1        = nn.Conv2d(num_inputs, self.num_features, 1, padding=0)
        self.conv2        = nn.Conv2d(self.num_features, num_outputs, 3, padding=1)
        self.fcq_w        = nn.Linear(input_emb_size, self.num_features)
        self.fcShift1     = nn.Linear(num_outputs*14*14+input_emb_size, self.num_features)
        self.fcShift2     = nn.Linear(self.num_features, self.num_features)
        self.dropout      = dropout

    def forward(self, input, att, word):
        ## FC q
        word_W = F.dropout(word, self.dropout, training = self.training)
        weight = F.tanh(self.fcq_w(word_W)).view(-1,self.num_features,1,1)
        ## FC v
        v = F.dropout2d(input, self.dropout, training = self.training)
        v = v * F.relu(1 - att).unsqueeze(1).expand_as(input)
        v = F.tanh(self.conv1(v))
        ## attMap
        inputAttShift = F.tanh(self.fcShift1(torch.cat((att.view(-1, self.num_outputs * 14 * 14),word), 1)))
        inputAttShift = F.tanh(self.fcShift2(inputAttShift)).view(-1, self.num_features, 1, 1)
        ## v * q_tile
        v = v * weight.expand_as(v) * inputAttShift.expand_as(v)
        # no tanh shoulb be here
        v = self.conv2(v)
        # Normalize to single area
        return F.softmax(v.view(-1, 14 * 14), dim=1).view(-1, self.num_outputs, 14, 14)

class NodeBlockResidual(nn.Module):
    def __init__(self, height, num_inputs, num_outputs, input_emb_size, attNum, dropout=0.3):
        super(NodeBlockResidual, self).__init__()
        self.featMapH     = 14
        self.featMapW     = 14
        self.num_proposal = 14*14
        self.img_emb_size = num_inputs
        self.attNum       = attNum
        self.height       = height
        self.dropout      = dropout

        # height = 1
        if height == 0:
            self.attImg = MLB(num_inputs,attNum,input_emb_size)
            self.bilnear = Bilnear(num_inputs*attNum, num_outputs, input_emb_size, dropout)
        else:
            self.attImgSon = AttImgSonModule(num_inputs,attNum,input_emb_size,self.dropout)
            self.bilnear = Bilnear(num_inputs * attNum + num_outputs, num_outputs, input_emb_size, dropout)

    def forward(self, sons, words, nodes, que_enc, attModule=None):
        inputImg = sons[2]
        if self.height == 0:
            attMap   = self.attImg(inputImg, words)
            feat     = attMap.view(-1, self.attNum, self.num_proposal).bmm(inputImg.view(-1, self.img_emb_size, self.num_proposal).transpose(1, 2)).view(-1, self.attNum * self.img_emb_size)
            ans_feat = self.bilnear(feat, que_enc)
            return (ans_feat, attMap)
        else:
            ## son input
            inputComFeat = sons[0].sum(1).squeeze(1)
            inputAtt     = sons[1].sum(1).squeeze()
            inputResFeat = sons[3].sum(1).squeeze(1)
            if inputAtt.dim() == 2: inputAtt = inputAtt.unsqueeze(0)
            ## cal att Map
            attMap = self.attImgSon(inputImg, inputAtt, words)
            ## x_t + son
            imFeat = attMap.view(-1, self.attNum, self.num_proposal).bmm(inputImg.view(-1, self.img_emb_size, self.num_proposal).transpose(1,2)).view(-1, self.attNum*self.img_emb_size)
            feat = torch.cat((inputComFeat, imFeat), 1) #feat = inputComFeat + imFeat#

            x_res = self.bilnear(feat, que_enc)
            ## up fused feat
            outResFeat = inputResFeat + x_res
            return (outResFeat, attMap)