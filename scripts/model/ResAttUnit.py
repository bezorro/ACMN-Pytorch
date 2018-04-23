import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

class Bilnear(nn.Module):

    def __init__(self, num_inputs, num_outputs, input_emb_size, dropout=0.3):
        super(Bilnear, self).__init__()
        self.num_features = 2048
        self.num_outputs = num_outputs
        self.fcv = nn.Linear(num_inputs, self.num_features)
        self.bnv = nn.BatchNorm1d(self.num_features)
        self.fcq = nn.Linear(input_emb_size, self.num_features)
        self.fco = nn.Linear(self.num_features, num_outputs)
        self.dropout = dropout

    def forward(self, input, word):
        ## FC q
        word_W = F.dropout(word, self.dropout, training = self.training)
        w = self.fcq(word_W)
        ## FC v
        v = F.dropout(input, self.dropout, training = self.training)
        v = self.fcv(v)
        # v = self.bnv(v)
        ## v * q_tile
        o = v * w
        o = F.relu(o)
        # no tanh shoulb be here
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
        weight = F.tanh(self.fcq_w(word_W)).view(-1,self.num_features,1,1)
        ## FC v
        v = input#F.dropout2d(input, self.dropout, training = self.training)
        v = F.tanh(self.conv1(v))
        ## v * q_tile
        v = v * weight.expand_as(v) # v = self.cbn1(F.tanh(v),word) #apply non-linear before cbn equal to MLB
        # no tanh shoulb be here
        v = self.conv2(v)
        return F.softmax(v.view(-1,14*14), dim=1).view(-1,self.num_outputs,14,14)

class AttImgSonModule(nn.Module):
    def __init__(self, num_inputs, num_outputs, input_emb_size, dropout=0.3):
        super(AttImgSonModule, self).__init__()
        self.num_features = 2048
        self.num_outputs = num_outputs
        self.num_inputs = num_inputs
        self.conv1 = nn.Conv2d(num_inputs, self.num_features, 1, padding=0)
        self.conv2 = nn.Conv2d(self.num_features, num_outputs, 3, padding=1)
        self.fcq_w = nn.Linear(input_emb_size, self.num_features)
        self.fcShift1 = nn.Linear(num_outputs*14*14+input_emb_size, self.num_features)
        self.fcShift2 = nn.Linear(self.num_features, self.num_features)
        self.dropout = dropout

    def forward(self, input, att, word):
        ## FC q
        word_W = F.dropout(word, self.dropout, training = self.training)
        weight = F.tanh(self.fcq_w(word_W)).view(-1,self.num_features,1,1)
        ## FC v
        v = F.dropout2d(input, self.dropout, training = self.training)
        v = v * F.relu(1-att).unsqueeze(1).expand_as(input)
        v = F.tanh(self.conv1(v))
        ## attMap
        inputAttShift = F.tanh(self.fcShift1(torch.cat((att.view(-1,self.num_outputs*14*14),word),1)))
        inputAttShift = F.tanh(self.fcShift2(inputAttShift)).view(-1,self.num_features,1,1)
        ## v * q_tile
        v = v * weight.expand_as(v) * inputAttShift.expand_as(v) # v = self.cbn1(F.tanh(v),word) #apply non-linear before cbn equal to MLB
        # no tanh shoulb be here
        v = self.conv2(v)
        # Normalize to single area
        return F.softmax(v.view(-1,14*14), dim=1).view(-1,self.num_outputs,14,14)

class NodeBlock(nn.Module):
    def __init__(self, height, num_inputs, num_outputs, input_emb_size, attNum, share_attModule=False, dropout=0.3):
        super(NodeBlock, self).__init__()
        self.featMapH = 14
        self.featMapW = 14
        self.num_proposal = 14*14
        self.img_emb_size = num_inputs
        self.attNum = attNum
        self.height = height
        self.dropout = dropout
        self.share_attModule = share_attModule

        # height = 1
        if height == 0:
            if not share_attModule: self.attImg = MLB(num_inputs,attNum,input_emb_size) # self.attImg.conv1.register_backward_hook(printgradnorm)
            self.g_fcim1 = nn.Linear(num_inputs*attNum+input_emb_size, num_outputs)
            self.g_fc2 = nn.Linear(num_outputs, num_outputs)
        else:
            # self.attImgSon = MLB(num_inputs+2*attNum,attNum,input_emb_size,self.dropout)
            if not share_attModule: self.attImgSon = AttImgSonModule(num_inputs,attNum,input_emb_size,self.dropout)
            self.g_fc1 = nn.Linear(num_inputs*attNum+num_outputs*height+input_emb_size, (num_inputs*attNum+num_outputs*height+input_emb_size)//2)
            self.g_fc2 = nn.Linear((num_inputs*attNum+num_outputs*height+input_emb_size)//2, num_outputs)

        self.g_fc3 = nn.Linear(num_outputs, num_outputs)
        self.g_fc4Res = nn.Linear(num_outputs, num_outputs)
        self.g_fc4Cat = nn.Linear(num_outputs, num_outputs)

    def forward(self, sons, words, nodes, que_enc, attModule=None):
        inputImg = sons[3]
        if self.height == 0:
            if self.share_attModule:
                inputAtt = Variable(torch.FloatTensor(inputImg.size(0), self.attNum, self.featMapH, self.featMapW).zero_()).cuda().squeeze()
                attMap = attModule(inputImg,inputAtt,words)
            else: attMap = self.attImg(inputImg, words)
            feat = attMap.view(-1, self.attNum, self.num_proposal).bmm(inputImg.view(-1,self.img_emb_size,self.num_proposal).transpose(1,2)).view(-1, self.attNum*self.img_emb_size)
            x_ = torch.cat((feat,que_enc),1)
            x_ = self.g_fcim1(x_)
            x_ = F.relu(x_)
            x_ = self.g_fc2(x_)
            x_ = F.relu(x_)
            x_ = self.g_fc3(x_)
            x_ = F.relu(x_)
            x_ = self.g_fc4Res(x_)
            ans_feat = F.relu(x_)
            return (ans_feat,ans_feat,attMap)
        else:
            ## son input
            inputResFeat = sons[0].sum(1).squeeze(1)
            if self.height > 1: inputCatFeat = sons[1].sum(1).squeeze(1)
            else: inputCatFeat = None
            # inputAtt = torch.cat((sons[2][:,0,:,:],sons[2][:,1,:,:]),1)
            inputAtt = sons[2].sum(1).squeeze()
            if inputAtt.dim() == 2: inputAtt = inputAtt.unsqueeze(0)
            ## cal att Map
            # attMap = self.attImgSon(torch.cat((inputImg,inputAtt),1), words)
            if self.share_attModule: attMap = attModule(inputImg,inputAtt,words)
            else: attMap = self.attImgSon(inputImg,inputAtt,words)
            ## x_t + son
            imFeat = attMap.view(-1, self.attNum, self.num_proposal).bmm(inputImg.view(-1,self.img_emb_size,self.num_proposal).transpose(1,2)).view(-1, self.attNum*self.img_emb_size)
            if inputCatFeat is None: feat = torch.cat((inputResFeat, imFeat), 1)
            else: feat = torch.cat((inputResFeat, inputCatFeat, imFeat), 1)

            x_ = self.g_fc1(torch.cat((feat, que_enc), 1))
            x_ = F.relu(x_)
            x_ = self.g_fc2(x_)
            x_ = F.relu(x_)
            x_ = self.g_fc3(x_)   ## g_fc3 can turn into LSTM
            x_ = F.relu(x_)
            x_res = self.g_fc4Res(x_)
            x_res = F.relu(x_res)
            x_cat = self.g_fc4Cat(x_)
            x_cat = F.relu(x_cat) 
            ## up fused feat
            outResFeat = inputResFeat+x_res # apply tanh if share weight
            if inputCatFeat is None: outCatFeat = x_cat
            else: outCatFeat = torch.cat((inputCatFeat, x_cat), 1) #or cat imFeat directly? if cat imFeat, attend single word? all node should contribute to final prediction
            return (outResFeat,outCatFeat,attMap)

class NodeBlockResidual(nn.Module):
    def __init__(self, height, num_inputs, num_outputs, input_emb_size, attNum, share_attModule=False, dropout=0.3):
        super(NodeBlockResidual, self).__init__()
        # assert(num_inputs == num_outputs)
        self.featMapH = 14
        self.featMapW = 14
        self.num_proposal = 14*14
        self.img_emb_size = num_inputs
        self.attNum = attNum
        self.height = height
        self.dropout = dropout
        self.share_attModule = share_attModule

        # height = 1
        if height == 0:
            if not share_attModule: self.attImg = MLB(num_inputs,attNum,input_emb_size) # self.attImg.conv1.register_backward_hook(printgradnorm)
            self.bilnear = Bilnear(num_inputs*attNum, num_outputs, input_emb_size, dropout)
            # self.g_fcim1 = nn.Linear(num_inputs*attNum+input_emb_size, 1024)
        else:
            # self.attImgSon = MLB(num_inputs+2*attNum,attNum,input_emb_size,self.dropout)
            if not share_attModule: self.attImgSon = AttImgSonModule(num_inputs,attNum,input_emb_size,self.dropout)
            self.bilnear = Bilnear(num_inputs*attNum+num_outputs, num_outputs, input_emb_size, dropout)
            # self.g_fc1 = nn.Linear(num_inputs*attNum+num_outputs+input_emb_size, 1024)

        # self.g_fc2 = nn.Linear(1024, num_outputs)
        # self.g_fc3 = nn.Linear(num_outputs, num_outputs)
        # self.g_fc4Res = nn.Linear(num_outputs, num_outputs)

    def forward(self, sons, words, nodes, que_enc, attModule=None):
        inputImg = sons[2]
        if self.height == 0:
            if self.share_attModule:
                inputAtt = Variable(torch.FloatTensor(inputImg.size(0), self.attNum, self.featMapH, self.featMapW).zero_()).cuda().squeeze()
                attMap = attModule(inputImg,inputAtt,words)
            else: attMap = self.attImg(inputImg, words)
            feat = attMap.view(-1, self.attNum, self.num_proposal).bmm(inputImg.view(-1,self.img_emb_size,self.num_proposal).transpose(1,2)).view(-1, self.attNum*self.img_emb_size)
            ans_feat = self.bilnear(feat,que_enc)
            # x_ = torch.cat((feat,que_enc),1)
            # x_ = self.g_fcim1(x_)
            # x_ = F.relu(x_)
            # x_ = self.g_fc2(x_)
            # x_ = F.relu(x_)
            # x_ = self.g_fc3(x_)
            # x_ = F.relu(x_)
            # x_ = self.g_fc4Res(x_)
            # ans_feat = F.relu(x_)
            return (ans_feat,attMap)
        else:
            ## son input
            inputComFeat = sons[0].sum(1).squeeze(1)
            inputAtt = sons[1].sum(1).squeeze()
            inputResFeat = sons[3].sum(1).squeeze(1)
            if inputAtt.dim() == 2: inputAtt = inputAtt.unsqueeze(0)
            ## cal att Map
            if self.share_attModule: attMap = attModule(inputImg,inputAtt,words)
            else: attMap = self.attImgSon(inputImg,inputAtt,words)
            ## x_t + son
            imFeat = attMap.view(-1, self.attNum, self.num_proposal).bmm(inputImg.view(-1,self.img_emb_size,self.num_proposal).transpose(1,2)).view(-1, self.attNum*self.img_emb_size)
            feat = torch.cat((inputComFeat, imFeat), 1)#feat = inputComFeat + imFeat#

            x_res = self.bilnear(feat,que_enc)
            # x_ = self.g_fc1(torch.cat((feat, que_enc), 1))
            # x_ = F.relu(x_)
            # x_ = self.g_fc2(x_)
            # x_ = F.relu(x_)
            # x_ = self.g_fc3(x_)   ## g_fc3 can turn into LSTM
            # x_ = F.relu(x_)
            # x_res = self.g_fc4Res(x_)
            # x_res = F.relu(x_res)
            ## up fused feat
            outResFeat = inputResFeat+x_res#outResFeat = inputComFeat+x_res # apply tanh if share weight
            return (outResFeat,attMap)