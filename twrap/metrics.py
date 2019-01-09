# @Author: Narsi Reddy <cibitaw1>
# @Date:   2018-09-22T17:38:05-05:00
# @Email:  sainarsireddy@outlook.com
# @Last modified by:   narsi
# @Last modified time: 2019-01-03T22:50:40-06:00
import torch
torch.manual_seed(29)
from torch import nn
import numpy as np
np.random.seed(29)
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.nn.parameter import Parameter
from math import exp

"""
CLASSIFICATION METRICS
"""
def accuracy(output, target):
    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
    res = correct_k.mul_(100.0 / batch_size)
    return res

def topK_accuray(k = 2):
    def accuracy(output, target):
        batch_size = target.size(0)

        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

        return correct_k.mul_(100.0 / batch_size)
    return accuracy

def binary_accuracy(output, target):

    res = torch.mean(target.eq(torch.round(output)).float()) * 100
    return res

"""
SUPER RESOLUTION
"""
# https://github.com/Po-Hsun-Su/pytorch-ssim
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel, sigma = 1.5):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size = 5, channel = 24, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = create_window(window_size, self.channel, sigma = 5)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, sigma = 5)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class SSIM_LOSS(nn.Module):
    def __init__(self, window_size = 5, channel = 1,size_average = True):
        super(SSIM_LOSS, self).__init__()
        self.SSIM = SSIM(window_size, channel, size_average)

    def forward(self, img1, img2):
        return 1-self.SSIM(img1, img2)

def psnr(output, target):
    mse = F.mse_loss(output, target)
    return -10. * logX(mse)
def logX(x, d = 10.0):
    """ Log10: log base 10 for tensorflow
    """
    numerator =  torch.log(x)
    denominator =  np.log(d)
    return numerator / denominator
