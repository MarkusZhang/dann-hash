"""
Complex model for training with classifier loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np

def flatten(tensor):
    return tensor.view(tensor.data.size(0),-1)

def _calc_output_dim(models,input_dim):
    """
    :param models: a list of model objects
    :param input_dim: like [3,100,100]
    :return:  the output dimension (in one number) if an image of `input_dim` is passed into `models`
    """
    input_tensor = torch.from_numpy(np.zeros(input_dim))
    input_tensor.unsqueeze_(0)
    img = Variable(input_tensor).float()
    output = img
    for model in models:
        output = model(output)
    return output.data.view(output.data.size(0), -1).size(1)


class BasicFeatExtractor(nn.Module):
    def __init__(self,params):
        super(BasicFeatExtractor, self).__init__()
        self.use_dropout = params.use_dropout
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self,x):
        conv1_out = self.conv1(x)
        return conv1_out

class CodeGen(nn.Module):
    "the size of the shared feature is the same as hash code"
    def __init__(self,params):
        super(CodeGen, self).__init__()
        self.use_dropout = params.use_dropout

        #### left part #####
        ####################
        self.left_conv1 = nn.Sequential(
            # first convolution
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.left_conv2 = nn.Sequential(
            # second convolution
            nn.Conv2d(32, 80, kernel_size=5, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        gen_output_dim = _calc_output_dim(models=[BasicFeatExtractor(params=params),self.left_conv1,self.left_conv2],
                                          input_dim=[3,params.image_scale,params.image_scale])

        # this size can be checked using shared_feat.data.size(1)
        self.left_l1 = nn.Linear(in_features=gen_output_dim, out_features=200)
        self.left_l1_bnm = nn.BatchNorm1d(200)
        self.left_l2 = nn.Linear(in_features=200, out_features=int(params.hash_size/2))

        #### right part #####
        ####################
        self.right_conv1 = nn.Sequential(
            # first convolution
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.right_conv2 = nn.Sequential(
            # second convolution
            nn.Conv2d(32, 80, kernel_size=5, padding=2),
            nn.BatchNorm2d(80),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        gen_output_dim = _calc_output_dim(models=[BasicFeatExtractor(params=params), self.right_conv1, self.right_conv2],
                                          input_dim=[3, params.image_scale, params.image_scale])

        # this size can be checked using shared_feat.data.size(1)
        self.right_l1 = nn.Linear(in_features=gen_output_dim, out_features=200)
        self.right_l1_bnm = nn.BatchNorm1d(200)
        self.right_l2 = nn.Linear(in_features=200, out_features=int(params.hash_size / 2))

        self.clf = nn.Linear(in_features=params.hash_size,out_features=10)

    def forward(self,x):
        # forward left part
        left_conv1_out = self.left_conv1(x)
        left_conv2_out = flatten(self.left_conv2(left_conv1_out))
        left_l1_out = F.sigmoid(self.left_l1_bnm(self.left_l1(left_conv2_out)))
        left_final_out = F.tanh(self.left_l2(left_l1_out))

        # forward right part
        right_conv1_out = self.right_conv1(x)
        right_conv2_out = flatten(self.right_conv2(right_conv1_out))
        right_l1_out = F.sigmoid(self.right_l1_bnm(self.right_l1(right_conv2_out)))
        right_final_out = F.tanh(self.right_l2(right_l1_out))

        # 16-bit code
        code = torch.cat([left_final_out,right_final_out],1)
        return code,F.softmax(self.clf(code))



class Discriminator(nn.Module):
    "produce a softmax label prediction"
    def __init__(self,params):
        super(Discriminator, self).__init__()
        input_size = _calc_output_dim(models=[BasicFeatExtractor(params=params)],
                                          input_dim=[3, params.image_scale, params.image_scale])

        self.l1 = nn.Linear(in_features=input_size,out_features=2)

    def forward(self,x):
        return F.softmax(self.l1(flatten(x)))


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # We return as many input gradients as there were arguments.
        # Gradients of non-Tensor arguments to forward must be None.
        return -grad_output