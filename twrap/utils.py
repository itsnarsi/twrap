# @Author: Narsi Reddy <cibitaw1>
# @Date:   2018-09-22T17:11:54-05:00
# @Email:  sainarsireddy@outlook.com
# @Last modified by:   cibitaw1
# @Last modified time: 2018-09-22T17:33:23-05:00
import os
import torch

from collections import OrderedDict
from terminaltables import DoubleTable

import numpy as np
import warnings
from torch.optim.optimizer import Optimizer

from sklearn.metrics import auc
import numbers

import matplotlib.pyplot as plt


"""
MODEL SUMMARY CODE
"""
def model_summary(input_var, model):
    # Adopted from https://github.com/pytorch/pytorch/issues/2001#issuecomment-313735757
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())[1:]
            summary[m_key]['output_shape'] = list(output.size())[1:]

            params = torch.tensor(0)
            if hasattr(module, 'weight') and module.weight is not None:
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            summary[m_key]['nb_mac_params'] = params.item()
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params.item()


        if not isinstance(module, torch.nn.Sequential) and \
           not isinstance(module, torch.nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []
    model.eval()
    # register hook
    model.apply(register_hook)
    # make a forward pass
    if type(input_var) is not list:
        model(input_var)
    else:
        model(*input_var)
    # remove these hooks
    for h in hooks:
        h.remove()

    # Block Starting point locations
    block_locs = []
    for layer in list(summary.keys()):
        if 'BLOCK' in layer:
            if block_locs == []:
                y = 1
                prev_loc = []
            else:
                y = int(prev_loc)+1
            prev_loc = layer.split('-')[-1]
            block_locs.append([layer, y])
    # Preparing Summary table
    summary_str = [['Block Name','Layer Name', 'Input Shape', 'Output  Shape', 'Param', 'ops']]
    block_summary_str = [['Layer Name', 'Input Shape', 'Output  Shape', 'Param', 'ops']]
    total_num_parm = 0
    total_num_ops = 0
    for layer in list(summary.keys()):
        if 'BLOCK' in layer:
            layer_info = summary[layer]
            layer_summary = [layer, str(layer_info['input_shape']), str(layer_info['output_shape']), str(layer_info['nb_params'])]
            block_summary_str.append(layer_summary)
        else:
            x = int(layer.split('-')[-1])
            block_name = ''
            for i in range(len(block_locs)):
                if x == block_locs[i][1]:
                    block_name = block_locs[i][0]

            layer_info = summary[layer]
            if 'conv2d' in layer.lower():
                num_ops = np.prod(np.array(layer_info['output_shape'][1:]+[layer_info['nb_mac_params']]))/1000000
            elif 'Linear' in layer:
                num_ops = np.prod(np.array([layer_info['nb_mac_params']]))/1000000
            else:
                num_ops = 0
            layer_summary = [block_name, layer, str(layer_info['input_shape']), str(layer_info['output_shape']), str(layer_info['nb_params']), num_ops]
            summary_str.append(layer_summary)
        total_num_parm += layer_info['nb_params']
        total_num_ops += num_ops
    # Printing Table
    table = DoubleTable(summary_str)
    block_table = DoubleTable(block_summary_str)

    print('================== Block Level ====================')
    print(block_table.table)
    print('\n')
    print('================== Layer Level ====================')
    print(table.table)
    print('Total number of Param:' + str(total_num_parm))
    print('Total number of Ops in Million:' + str(total_num_ops))

    return (summary, total_num_parm, total_num_ops)

"""
TO CATEGORICAL
"""
def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical

"""
GET FILES
"""
def get_filepaths(directory,file_type=''):
    # file_type should be tuple to match for multiple extentions
    file_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            if filepath.endswith(file_type):
                file_paths.append(filepath)
    return file_paths

"""
PYTORCH MODEL PARAMETER FILTERS
"""
# Filtering out weights with gradients
def parfilter(model):
    return filter(lambda p: p.requires_grad, model.parameters())
# Non - Trainable Model
def non_trainable_model(model):
    for param in model.parameters():
        param.requires_grad = False
# Trainable Model
def trainable_model(model):
    for param in model.parameters():
        param.requires_grad = True
# Block Specific layer non-trainable
def non_trainable_layer(layers):
    if type(layers) is list:
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False
    else:
        for param in layers.parameters():
            param.requires_grad = False
# Block Specific layer trainable
def trainable_layer(layers):
    if type(layers) is list:
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = True
    else:
        for param in layers.parameters():
            param.requires_grad = True

"""
MATCHING SCORE METRICS
"""
def genROC(traget, score, resolution = 10000, gmrsat = [0.01, 0.001, 0.0001]):

    T = score[traget==1]

    F = score[traget==0]
    min_score = np.min(score)
    max_score = np.max(score)
    inc = np.abs(max_score - min_score)/resolution
    bins_ = np.arange(min_score-inc, max_score+inc, inc)
    T1 = np.histogram(T, bins = list(bins_))
    T1_bins = T1[1][0:len(bins_)-1]
    TPR = np.cumsum(T1[0])/np.sum(T1[0])

    F1 = np.histogram(F, bins = list(bins_))
    F1_bins = F1[1][0:len(bins_)-1]
    FPR = np.cumsum(F1[0])/np.sum(F1[0])

    thresholds = bins_

    TAR = []
    FAR = []
    for i in range(len(thresholds)):
        t = thresholds[i]
        t_t = TPR[T1_bins <= t];
        f_t = FPR[F1_bins <= t]
        if len(f_t) > 0 and len(t_t) > 0:
            f_t = np.max(f_t);
            t_t = np.max(t_t);
            TAR.append(t_t)
            FAR.append(f_t)

    TAR = np.asarray(TAR)
    FAR = np.asarray(FAR)

    EER = FAR[np.nanargmin(np.absolute(FAR-1+TAR))]
    EER_thresh = thresholds[np.nanargmin(np.absolute(FAR-1+TAR))]
    AUC = auc(FAR, TAR)


    GMRS= []
    for i in gmrsat:
        try:
            gms = np.max(TAR[FAR <= i])
        except:
            gms = 0.0
        GMRS.append(gms * 100)

    return (TAR, FAR, thresholds, EER, EER_thresh, AUC, GMRS)
