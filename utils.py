import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.modules.loss import _Loss
import math

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight, gain=1)
        # init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias, 0)

def params_count(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class mean_squared_error(_Loss):
    def __init__(self):
        super(mean_squared_error, self).__init__()

    def forward(self, input, target):
        # _assert_no_grad(target)
        return torch.nn.functional.mse_loss(input, target)

def get_full_name_list(list_name, stage_flag):
    full_image_list = []
    with open(list_name) as f:
        image_list = f.readlines()  # 00001/0001　-> 0001
    if stage_flag == 'train':
        str_print = 'len(train_image_list):'
    elif stage_flag == 'test':
        str_print = 'len(test_image_list):'
    print(str_print + str(len(image_list)))
    for i in range(0, len(image_list)):
        if stage_flag == 'train':
            if i % 2 == 0:
                til = image_list[i].rstrip()
                for j in range(1, 8):
                    til_png = til + '/im' + str(j) + '.png'
                    full_image_list.append(til_png)
        elif stage_flag == 'test':
            til = image_list[i].rstrip()
            til_png = til + '/im4.png'
            full_image_list.append(til_png)

    if stage_flag == 'train':
        str_print = 'len(full_train_image_list):'
    elif stage_flag == 'test':
        str_print = 'len(full_test_image_list):'
    print(str_print + str(len(full_image_list)))
    return full_image_list

def calculate_psnr(input, prediction, label):
    psnrs = []
    mse_pre = math.sqrt(torch.mean((prediction - label) ** 2.0))
    psnr_pre = 20 * math.log10(1.0/mse_pre)
    mse_input = math.sqrt(torch.mean((input - label) ** 2.0))
    psnr_input = 20 * math.log10(1.0/mse_input)
    psnrs.append(psnr_pre)
    psnrs.append(psnr_input)
    return psnrs

def calculate_psnr_single(prediction, label):
    mse_pre = math.sqrt(torch.mean((prediction - label) ** 2.0))
    psnr_pre = 20 * math.log10(1.0/mse_pre)
    return psnr_pre
