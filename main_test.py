import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from dataset import *
from utils import *
from model import *

import random

QT_FACTOR = 40
# Params
parser = argparse.ArgumentParser(description='PyTorch ARCNN/FastARCNN/DnCNN')
parser.add_argument('--model_choice', default='DnCNN', type=str, help='choose the model type')
parser.add_argument('--QT_FACTOR', default=40, type=int, help='quantization factor')
parser.add_argument('--test_batchsize', default=8, type=int, help='training batches')
parser.add_argument('--InputFolder', default='/home/zhouyi/new/ARCNN-pytorch/data/vimeo_part_q40_crop', type=str, help='inputfolder')
parser.add_argument('--LabelFolder', default='/home/zhouyi/new/ARCNN-pytorch/data/vimeo_part_crop', type=str, help='LabelFolder')
parser.add_argument('--model_dir', default='/home/zhouyi/new/ARCNN-pytorch/models/DnCNN_q40/checkpoint_q40_030.pth.tar', type=str, help='model path')
parser.add_argument('--Testlist', default='/home/zhouyi/new/ARCNN-pytorch/data/temp_sep_validationlist.txt', type=str,
                    help='testlist')  # 7824

if __name__ == '__main__':
    args = parser.parse_args()

    print('args:++++++++++++++++++++++')
    for k, v in sorted(vars(args).items()):
        print(str(k) + ": " + str(v))
    print('+++++++++++++++++++++++++++')

    QT_FACTOR = args.QT_FACTOR

    # build model
    print('===> Building model')
    if args.model_choice == 'ARCNN':
        model = ARCNN()
    elif args.model_choice == 'FastARCNN':
        model = FastARCNN()
    elif args.model_choice == 'DnCNN':
        model = DnCNN()
    else:
        assert False, 'ERROR: no specific model choice !!!'

    # if use GPU
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cuda:
        model = model.cuda()
        model.to(device)

    # generate test list
    stage_flag = 'test'
    full_test_image_list = get_full_name_list(args.Testlist, stage_flag)
    test_nums = len(full_test_image_list)
    batch_size_test = args.test_batchsize
    full_test_batchsize = batch_size_test
    n_iterations = int(test_nums / full_test_batchsize)

    # load model
    if not os.path.exists(args.model_dir):
        print('Cannot find model...')
    else:
        checkpoint = torch.load(args.model_dir)
        model.load_state_dict(checkpoint['state_dict'])

        print('load trained model...' + args.model_dir)
        print("model type: {}".format(type(model)))
        print(model)

    model.eval()

    print('start testing...')
    psnr_pre_list = []
    psnr_input_list = []
    psnr_diff_list = []

    for iteration in range(0, n_iterations):
        print('processing ' + str(iteration) + ' test_batch...')
        print('len(psnr_pre_list): ' + str(len(psnr_pre_list)))
        index_start = int(iteration * full_test_batchsize)
        index_end = int((iteration + 1) * full_test_batchsize)
        batch_paths = []
        for index in range(index_start, index_end):
            batch_paths.append(full_test_image_list[index])

        xs_test = generate_single_batch(train_input_dir=args.InputFolder, train_label_dir=args.LabelFolder,
                                        batch_paths=batch_paths, crop_flag=False)

        for i in range(0, len(xs_test)):
            xs_test[i] = (xs_test[i].astype('float32') / 255.0)
            xs_test[i] = torch.from_numpy(xs_test[i].transpose((0, 3, 1, 2)))  # torch.Size([7, 3, 256, 448])

        DDataset_test = ArtifactDataset(xs_test)
        DLoader_test = DataLoader(dataset=DDataset_test, num_workers=4, drop_last=False, batch_size=batch_size_test,
                                  shuffle=False)
        with torch.no_grad():
            for n_count, batch_yx in enumerate(DLoader_test):
                batch_x, batch_y = batch_yx[1], batch_yx[0]
                if cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()  # batch_x: label batch_y:input
                    batch_x.to(device)
                    batch_y.to(device)
                model_output = model(batch_y)
                for s in range(0, model_output.shape[0]):
                    psnrs = calculate_psnr(batch_y[s], model_output[s], batch_x[s])
                    psnr_pre = psnrs[0]
                    psnr_input = psnrs[1]
                    psnr_pre_list.append(psnr_pre)
                    psnr_input_list.append(psnr_input)
                    psnr_diff_list.append(psnr_pre - psnr_input)
                    print('Test: psnr_pre = %2.4f, psnr_input = %2.4f, psnr_diff = %2.4f ' % (
                        psnr_pre, psnr_input, abs(psnr_pre - psnr_input)))

    print('len(test_images): ' + str(len(psnr_pre_list)))
    print('mean_psnr_prediction: ' + str(np.mean(psnr_pre_list)))
    print('mean_psnr_input: ' + str(np.mean(psnr_input_list)))
    print('mean_psnr_diff: ' + str(np.mean(psnr_diff_list)))