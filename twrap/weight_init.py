# @Author: Narsi Reddy <narsi>
# @Date:   2018-11-12T14:06:36-06:00
# @Last modified by:   narsi
# @Last modified time: 2018-11-22T12:19:30-06:00
import torch
import torch.nn as nn
import torch.nn.init as init
'''
https://gist.github.com/jeasinema/ed9236ce743c8efaf30fa2ff732749f5
'''

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal(m.weight.data)
        try:
            init.normal(m.bias.data)
        except:
            pass
    elif isinstance(m, nn.Conv2d):
        init.xavier_uniform(m.weight.data , gain=np.sqrt(2))
        try:
            init.constant(m.bias.data, 0)
        except:
            pass
    elif isinstance(m, nn.Conv3d):
        init.xavier_uniform(m.weight.data , gain=np.sqrt(2))
        try:
            init.normal(m.bias.data)
        except:
            pass
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal(m.weight.data)
        try:
            init.normal(m.bias.data)
        except:
            pass
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_uniform(m.weight.data , gain=np.sqrt(2))
        try:
            init.normal(m.bias.data)
        except:
            pass
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_uniform(m.weight.data , gain=np.sqrt(2))
        try:
            init.normal(m.bias.data)
        except:
            pass
    elif isinstance(m, nn.BatchNorm1d):
        init.constant(m.weight.data, 1)
        try:
            init.constant(m.bias.data, 0)
        except:
            pass
    elif isinstance(m, nn.BatchNorm2d):
        try:
            init.constant(m.weight.data, 1)
        except:
            pass
        try:
            init.constant(m.bias.data, 0)
        except:
            pass
    elif isinstance(m, nn.BatchNorm3d):
        init.constant(m.weight.data, 1)
        try:
            init.constant(m.bias.data, 0)
        except:
            pass
    elif isinstance(m, nn.Linear):
        init.xavier_uniform(m.weight.data , gain=np.sqrt(2))
        try:
            init.constant(m.bias.data, 0)
        except:
            pass
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal(param.data)
            else:
                init.normal(param.data)


if __name__ == '__main__':
    pass
