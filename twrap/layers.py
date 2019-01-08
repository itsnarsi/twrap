# @Author: Narsi Reddy <cibitaw1>
# @Date:   2018-09-19T12:00:10-05:00
# @Email:  sainarsireddy@outlook.com
# @Last modified by:   narsi
# @Last modified time: 2018-11-22T23:24:26-06:00

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Function

np.random.seed(29)
torch.manual_seed(29)

"""
POOLING BLOCKS
"""
def pooling_brick(pooling_type, kernel_size, stride, padding, make_sequential):
    brick = []

    if padding == 'same':
        pad_size = int(np.floor(kernel_size/2))
    else:
        pad_size = 0

    # Pooling layers
    if pooling_type == 'max':
        brick.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=pad_size))
    elif pooling_type == 'avg':
        brick.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad_size))


    if make_sequential is True:
        brick = nn.Sequential(*brick)

    return brick


"""
CONVOLUTIONAL BLOCKS
"""

def conv2d_brick_1(in_ch, out_ch, kernel, stride, padding,
                   dilation, groups, bias, batch_norm, activation,
                   dropout, make_sequential, scale = 1.0):
    '''

    CONV BRICK Type - 1:

        DROPOUT -> CONV -> BATCH_NORM -> ACTIVATION

    '''
    brick = [] # Sequential layers list

    # Dropout
    if dropout > 0.0:
        brick.append(nn.Dropout(dropout))

    # Convoulutional layer padding
    pad_size = int(np.floor((kernel-1)/2))+int(dilation - 1)
    if padding == 'rep':
        brick.append(nn.ReflectionPad2d(pad_size))
        pad_size = 0
    elif padding == 'valid':
        pad_size = 0

    # Convolutional Layer
    brick.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad_size,
                                   bias = bias, stride=stride, dilation = dilation, groups=groups))

    # Batch Normalization
    if batch_norm is True:
        brick.append(nn.BatchNorm2d(out_ch))

    # Activation
    brick.append(ACTIVATIONS(activation))

    if make_sequential is True:
        brick = nn.Sequential(*brick)

    return brick

def conv2d_brick_2(in_ch, out_ch, kernel, stride, padding,
                   dilation, groups, bias, batch_norm, activation,
                   dropout, make_sequential, scale = 1.0):
    '''

    CONV BRICK Type - 2:

        BATCH_NORM -> ACTIVATION -> CONV -> DROPOUT

    '''
    brick = [] # Sequential layers list


    # Batch Normalization
    if batch_norm is True:
        brick.append(nn.BatchNorm2d(in_ch))

    # Activation
    brick.append(ACTIVATIONS(activation))

    # Convoulutional layer padding
    pad_size = int(np.floor((kernel-1)/2))+int(dilation - 1)
    if padding == 'rep':
        brick.append(nn.ReflectionPad2d(pad_size))
        pad_size = 0
    elif padding == 'valid':
        pad_size = 0

    # Convolutional Layer
    brick.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad_size,
                                   bias = bias, stride=stride, dilation = dilation, groups=groups))

    # Dropout
    if dropout > 0.0:
        brick.append(nn.Dropout(dropout))

    if make_sequential is True:
        brick = nn.Sequential(*brick)

    return brick

def conv2d_brick_3(in_ch, out_ch, kernel, stride, padding,
                   dilation, groups, bias, batch_norm, activation,
                   dropout, make_sequential, scale):
    '''

    CONV BRICK Type - 3:

        BATCH_NORM -> ACTIVATION -> CONVkxk_ch/scale_stride -> DROPOUT ->
        BATCH_NORM -> ACTIVATION -> CONVkxk_ch -> DROPOUT

    * Used in RESNET type - 1
    '''
    brick = [] # Sequential layers list

    if type(dropout) is not list:
        dropout = [dropout, dropout]

    brick += conv2d_brick_2(in_ch, int(out_ch/scale), kernel, stride, padding,
                   dilation, groups, bias, batch_norm, activation,
                   dropout[0], False)

    brick += conv2d_brick_2(int(out_ch/scale), out_ch, kernel, 1, padding,
                   dilation, groups, bias, batch_norm, activation,
                   dropout[1], False)

    if make_sequential is True:
        brick = nn.Sequential(*brick)

    return brick

def conv2d_brick_4(in_ch, out_ch, kernel, stride, padding,
                   dilation, groups, bias, batch_norm, activation,
                   dropout, make_sequential, scale):
    '''

    CONV BRICK Type - 4:

        BATCH_NORM -> ACTIVATION -> CONV1x1_ch/scale -> DROPOUT ->
        BATCH_NORM -> ACTIVATION -> CONVkxk_ch/scale_stride -> DROPOUT ->
        BATCH_NORM -> ACTIVATION -> CONV1x1_ch -> DROPOUT

    * Used in RESNET type - 2
    '''
    brick = [] # Sequential layers list

    if type(dropout) is not list:
        dropout = [dropout, dropout, dropout]

    brick += conv2d_brick_2(in_ch, int(out_ch/scale), 1, 1, padding,
                   dilation, 1, bias, batch_norm, activation,
                   dropout[0], False)

    brick += conv2d_brick_2(int(out_ch/scale), int(out_ch/scale), kernel, stride, padding,
                   dilation, int(groups), bias, batch_norm, activation,
                   dropout[1], False)

    brick += conv2d_brick_2(int(out_ch/scale), out_ch, 1, 1, padding,
                   dilation, 1, bias, batch_norm, activation,
                   dropout[2], False)

    if make_sequential is True:
        brick = nn.Sequential(*brick)

    return brick

def conv2d_brick_5(in_ch, out_ch, kernel, stride, padding,
                   dilation, groups, bias, batch_norm, activation,
                   dropout, make_sequential, scale):
    '''

    CONV BRICK Type - 5:

        BATCH_NORM -> ACTIVATION -> CONVkxk_ch/scale_stride -> DROPOUT ->
        BATCH_NORM -> ACTIVATION -> CONV1x1_ch -> DROPOUT

    * Used in RESNET type - 3
    '''
    brick = [] # Sequential layers list

    if type(dropout) is not list:
        dropout = [dropout, dropout]

    brick += conv2d_brick_2(in_ch, int(out_ch/scale), kernel, stride, padding,
                   dilation, int(groups), bias, batch_norm, activation,
                   dropout[0], False)

    brick += conv2d_brick_2(int(out_ch/scale), out_ch, 1, 1, padding,
                   dilation, 1, bias, batch_norm, activation,
                   dropout[0], False)

    if make_sequential is True:
        brick = nn.Sequential(*brick)

    return brick

class CONV2D_BLOCK(nn.Module):
    def __init__(self, kernel, in_ch, filters = [32], stride=1, padding='valid', activation='relu', use_bias=False,
                 dropout = 0.0, batch_norm = False, dilation = 1, groups = 1, conv_type = 1, scale = 4.0,
                 pool_type = 'max',pool_size = 2, pool_stride = 2, pool_padding = 'valid'):
        super(CONV2D_BLOCK, self).__init__()

        # Inputs
        if isinstance(filters, list) is False:
            raise NotImplementedError("filters should be in list, ex: filters = [conv1_channels,conv2_channels]")

        self.conv_type = conv_type

        self.in_ch = in_ch
        self.filters = [self.in_ch]+filters
        self.kernel = kernel
        self.stride = stride
        self.activation = activation
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.dropout = dropout
        self.dilation = dilation
        self.groups = groups
        self.pool_padding = pool_padding
        self.padding = padding
        self.scale = scale

        if self.conv_type == 1:
            self.brick = conv2d_brick_1
        elif self.conv_type == 2:
            self.brick = conv2d_brick_2
        elif self.conv_type == 3:
            self.brick = conv2d_brick_3
        elif self.conv_type == 4:
            self.brick = conv2d_brick_4
        elif self.conv_type == 5:
            self.brick = conv2d_brick_5

        self.layer = self.make_layer()

    def make_layer(self):
        node  = []
        for i in range(1, len(self.filters)):
            node += self.brick(self.filters[i-1], self.filters[i], self.kernel, self.stride, self.padding,
                   self.dilation, self.groups, self.use_bias, self.batch_norm, self.activation,
                   self.dropout, False, self.scale)

        if self.pool_type is not None:
            node += pooling_brick(self.pool_type, self.pool_size, self.pool_stride, self.pool_padding, False)

        return nn.Sequential(*node)

    def forward(self, input):
        return self.layer(input)

"""
RESIDUAL BLOCK
"""

class resnet_module_type1(nn.Module):
    def __init__(self, kernel, input_filter = 32, ouput_filters = 32, activation='relu', dropout = 0.0, use_bias = False,
                 stride = 1, padding = 'same', dilation = 1, groups = 1, scale = 1.0, batch_norm = True):
        super(resnet_module_type1, self).__init__()

        self.ouput_filters = ouput_filters
        self.input_filter = input_filter
        self.kernel = kernel
        self.activation = activation
        self.use_bias = use_bias
        self.stride = stride
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dilation = dilation
        self.groups = groups
        self.scale = scale
        self.padding = padding

        self.layer = self.make_layer()

    def make_layer(self):
        node = conv2d_brick_3(self.input_filter, self.ouput_filters, self.kernel, self.stride, self.padding,
                   self.dilation, self.groups, self.use_bias, self.batch_norm, self.activation,
                   self.dropout, True, self.scale)

        return node

    def forward(self, input):
        feat = self.layer(input)
        if self.stride == 1 and self.input_filter == self.ouput_filters:
            feat = feat+input
        return feat

class resnet_module_type2(nn.Module):
    def __init__(self, kernel, input_filter = 32, ouput_filters = 32, activation='relu', dropout = 0.0, use_bias = False,
                 stride = 1, padding = 'same', dilation = 1, groups = 1, scale = 4.0, batch_norm = True):
        super(resnet_module_type2, self).__init__()

        self.ouput_filters = ouput_filters
        self.input_filter = input_filter
        self.kernel = kernel
        self.activation = activation
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.stride = stride
        self.dropout = dropout
        self.dilation = dilation
        self.groups = groups
        self.scale = scale
        self.padding = padding

        self.layer = self.make_layer()

    def make_layer(self):
        node = conv2d_brick_4(self.input_filter, self.ouput_filters, self.kernel, self.stride, self.padding,
                   self.dilation, self.groups, self.use_bias, self.batch_norm, self.activation,
                   self.dropout, True, self.scale)

        return node

    def forward(self, input):
        feat = self.layer(input)
        if self.stride == 1 and self.input_filter == self.ouput_filters:
            feat = feat+input
        return feat


class RESNET_BLOCK(nn.Module):
    def __init__(self, kernel, in_ch, filters = [32], stride=1, padding='same', activation='relu', use_bias=False, dropout = 0.0,
                 batch_norm = False, groups = 1, dilation = 1, scale = 4.0, resnet_type = 1,
                 pool_type = 'max',pool_size = 2, pool_stride = 2, pool_padding = 'valid'):

        super(RESNET_BLOCK, self).__init__()
        # Inputs

        if isinstance(filters, list) is False:
            raise NotImplementedError("filters should be in list, ex: filters = [conv1_channels,conv2_channels]")

        self.in_ch = in_ch
        self.filters = [self.in_ch]+filters
        self.kernel = kernel
        self.stride = stride
        self.activation = activation
        self.use_bias = use_bias
        self.batch_norm = batch_norm
        self.pool_type = pool_type
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.dropout = dropout

        self.resnet_type = resnet_type

        self.groups = groups
        self.pool_padding = pool_padding
        self.dilation = dilation
        self.padding = padding
        self.scale = scale


        if self.resnet_type == 1:
            self.resnet_module = resnet_module_type1
        elif self.resnet_type == 2:
            self.resnet_module = resnet_module_type2

        self.layer = self.make_layer()

    def make_layer(self):
        node = []

        # Convolutional module
        for i in range(1, len(self.filters) - 1):
            node.append(self.resnet_module(self.kernel, input_filter = self.filters[i-1], ouput_filters = self.filters[i], activation=self.activation, dropout = self.dropout, use_bias = self.use_bias,
                                                 stride = 1, padding = self.padding, dilation = self.dilation, groups = self.groups, scale = self.scale, batch_norm = self.batch_norm))

        node.append(self.resnet_module(self.kernel, input_filter = self.filters[-2], ouput_filters = self.filters[-1], activation=self.activation, dropout = self.dropout, use_bias = self.use_bias,
                                             stride = self.stride, padding = self.padding, dilation = self.dilation, groups = self.groups, scale = self.scale, batch_norm = self.batch_norm))

        if self.pool_type is not None:
            node += pooling_brick(self.pool_type, self.pool_size, self.pool_stride, self.pool_padding, False)

        return nn.Sequential(*node)

    def forward(self, input):
        return self.layer(input)


"""
MLP(Multi-Layer Perceptron) BLOCK
"""

class MLP_BLOCK(nn.Module):
    def __init__(self,in_size, neurons = [32], activation='relu', use_bias=True, dropout = 0.0, batch_norm = False):

        super(MLP_BLOCK, self).__init__()
        self.in_size = in_size
        self.neurons = [self.in_size]+neurons
        self.activation = activation
        self.use_bias = use_bias
        self.dropout = dropout
        self.batch_norm = batch_norm

        if isinstance(self.neurons, list) is False:
            raise NotImplementedError("neurons should be in list, ex: filters = [FC1_neurons,FC2_neurons]")

        self.layer = self.make_layer()

    def make_layer(self):
        node = []

        # Dense module
        for i in range(1, len(self.neurons)):
            # Dropout
            if self.dropout > 0.0:
                node.append(nn.Dropout(self.dropout))
            # dense
            node.append(nn.Linear(self.neurons[i-1], self.neurons[i], bias = self.use_bias))

            # batchnorm
            if self.batch_norm is True:
                node.append(nn.BatchNorm1d(self.neurons[i]))

            # activation
            node.append(ACTIVATIONS(self.activation))

        return nn.Sequential(*node)

    def forward(self, input):
        return self.layer(input)

class norm_mlp_layer(nn.Module):
    def __init__(self, input, output):
        super(norm_mlp_layer, self).__init__()
        self.input = input
        self.output = output

        self.layer = nn.utils.weight_norm(nn.Linear(input, output, bias=False))
    def forward(self, input):
        # input = nn.functional.normalize(input)
        return self.layer(input)



"""
ACTIVATION LAYERS
"""
def ACTIVATIONS(name):
    if name == 'relu':
        return nn.ReLU(inplace=False)
    elif name == 'relu6':
        return nn.ReLU6(inplace=False)
    elif name == 'elu':
        return nn.ELU(inplace=False)
    elif name == 'selu':
        return nn.SELU(inplace=False)
    elif name == 'softmax':
        return nn.Softmax()
    elif name == 'softplus':
        return nn.Softplus()
    elif name == 'tanh':
        return nn.Hardtanh(-1, 1)
    elif name == 'sigm':
        return nn.Hardtanh(0, 1)
    elif name == 'linear':
        return linear()
    elif name == 'binthres':
        return ThresAct()
    elif name == 'l2norm':
        return L2_norm()
    elif name == 'clip':
        return clip()
    elif name == 'bintanh':
        return BinTANH()
    elif name == 'leakyrelu02':
        return nn.LeakyReLU(0.02)

def zmeanuvar(x):
    x_m = torch.mean(x, 2,keepdim = True)
    x_m = torch.mean(x_m, 3,keepdim = True)
    x = x - x_m
    x_s = torch.std(x, 2,keepdim = True)
    x_s = torch.std(x_s, 3,keepdim = True)
    x = x/(x_s+1e-5)
    return x

class clip(nn.Module):
    def __init__(self):
        super(clip, self).__init__()
    def forward(self, input):
        return input.clamp(min=0, max =1)

class L2_norm(nn.Module):
    def __init__(self):
        super(L2_norm, self).__init__()
    def forward(self, input):
        return F.normalize(input)

class linear(nn.Module):
    def __init__(self):
        super(linear, self).__init__()
    def forward(self, x):
        return x


class thresAct(torch.autograd.Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        c = input.clamp(min=0, max =1)
        c[c > 0] = 1
        return c
    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 1] = 0
        return grad_input, None

class ThresAct(torch.nn.Module):

    def __init__(self):
        super(ThresAct, self).__init__()

        self.quantclip = thresAct

    def forward(self, input):
        return self.quantclip.apply(input)


class binTanH(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """
    @staticmethod
    def forward(self, input):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        self.save_for_backward(input)
        c = input.clamp(min=-1, max =1)
        c[c > 0.25] = 1
        c[c > -0.25] = -1
        c[( 1 * (c <= 0.25) * 1 * (c >= -0.25)) == 1] = 0
        return c
    @staticmethod
    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < -1] = 0
        grad_input[input > 1] = 0
        return grad_input, None

class BinTANH(torch.nn.Module):

    def __init__(self):
        super(BinTANH, self).__init__()

        self.quantclip = binTanH

    def forward(self, input):
        return self.quantclip.apply(input)


"""
MISC FUNCTIONS
"""

def flatten(input):
    return input.view(input.size(0), -1)

class PWD_LAYER(nn.Module):
    #Pair Wise Destance 2D
    def __init__(self):
        super(PWD_LAYER, self).__init__()

    def forward(self, x,y):
        batch_size = x.size(0)
        feat_size = x.size(1)
        h = x.size(2)
        w = x.size(3)
        x = x.view(batch_size, feat_size, -1)
        y = y.view(batch_size, feat_size, -1)
        bs, points_dim, num_points = x.size()
        xx = torch.bmm(x.transpose(2,1), x)
        yy = torch.bmm(y.transpose(2,1), y)
        zz = torch.bmm(y.transpose(2,1), x)
        diag_ind = torch.arange(0, num_points).type(torch.LongTensor)
        if x.is_cuda:
            diag_ind = diag_ind.cuda(x.get_device())
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2,1) + ry - 2*zz)

        P = P.view(batch_size, w*h, h, w)
        return P
