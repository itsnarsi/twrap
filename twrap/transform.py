# @Author: Narsi Reddy <narsi>
# @Date:   2018-11-12T14:19:49-06:00
# @Last modified by:   narsi
# @Last modified time: 2019-06-02T12:30:12-05:00

import torch
from PIL import Image, ImageFilter
import numpy as np

class Tensor2PIL(object):

    def __call__(self, I):

        I = I.data.cpu().numpy().copy()
        return Image.fromarray(I)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToZNorm(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, I):

        I = np.float32(I).copy()
        I = I/255.0
        if len(I.shape) == 2:
            I = (I - np.mean(I))/(np.std(I)+0.0001)
            I = np.expand_dims(I, 0)
        elif len(I.shape) == 3:
            I[..., 0] = (I[..., 0] - np.mean(I[..., 0]))/(np.std(I[..., 0])+0.0001)
            I[..., 1] = (I[..., 1] - np.mean(I[..., 1]))/(np.std(I[..., 1])+0.0001)
            I[..., 2] = (I[..., 2] - np.mean(I[..., 2]))/(np.std(I[..., 2])+0.0001)
            I = np.transpose(I, (2, 0, 1))

        I = torch.from_numpy(I.copy())
        return I

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ToImg05Mean(object):

    def __call__(self, I):

        I = np.float32(I).copy()
        I = I/255.0
        if len(I.shape) == 2:
            I = np.expand_dims(I, 0)
        elif len(I.shape) == 3:
            I = np.transpose(I, (2, 0, 1))

        I = I - 0.5

        I = torch.from_numpy(I.copy())
        return I

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomResize(object):

    def __init__(self, min_scale = 0.8, max_scale = 1.2, interp = Image.BILINEAR):
        self.min = min_scale
        self.max = max_scale
        self.scale_diff = max_scale - min_scale
        self.interp = interp

    def __call__(self, I):

        scale = self.scale_diff * np.random.random_sample() + self.min

        width, height = I.size

        new_width = int(width * scale)
        new_height = int(height * scale)

        return I.resize((new_width, new_height), self.interp)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomBlur(object):

    def __init__(self, min_scale = 0.0, max_scale = 2):
        self.min = min_scale
        self.max = max_scale
        self.scale_diff = max_scale - min_scale

    def __call__(self, I):

        scale = self.scale_diff * np.random.random_sample() + self.min

        return I.filter(ImageFilter.GaussianBlur(scale))

    def __repr__(self):
        return self.__class__.__name__ + '()'
