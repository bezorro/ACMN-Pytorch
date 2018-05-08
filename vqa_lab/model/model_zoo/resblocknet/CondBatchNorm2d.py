# from torch.nn.modules.batchnorm import _BatchNorm
# from torch.nn.modules.module import Module
# from torch.nn.parameter import Parameter
# import torch
import torch.nn.functional as F
import torch.nn as nn

class CondBatchNorm2d(nn.Module):

    def __init__(self, num_features, num_wemb, eps=1e-5, momentum=0.1, dropout=0.0):
        super(CondBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.fcq_w = nn.Linear(num_wemb, num_features)
        #self.fcq_b = nn.Linear(num_wemb, num_features)

    def forward(self, input, word):
        weight = self.fcq_w(word).view(-1,self.num_features,1,1)# + 1
        #bias = self.fcq_b(word).view(-1,self.num_features,1,1)
        return input*weight.expand_as(input)#+bias.expand_as(input)


# class CondBatchNorm2d(_BatchNorm):

#     def __init__(self, num_features, num_wemb, eps=1e-5, momentum=0.1, affine=False):
#         super(CondBatchNorm2d, self).__init__(num_features, eps, momentum, False)
#         self.fcq_w = nn.Linear(num_wemb, num_features)
#         self.fcq_b = nn.Linear(num_wemb, num_features)

#     def forward(self, input, word):
#         weight = self.fcq_w(word)
#         bias = self.fcq_b(word)
#         output = super(CondBatchNorm2d, self).forward(input)
#         return output*weight+bias

#     def _check_input_dim(self, input):
#         if input.dim() != 4:
#             raise ValueError('expected 4D input (got {}D input)'
#                              .format(input.dim()))
#         super(CondBatchNorm2d, self)._check_input_dim(input)