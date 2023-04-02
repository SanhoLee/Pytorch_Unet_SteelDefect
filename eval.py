## necessary modules
import argparse
import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt

from dataset import *
from utils import *
from model import UNet

## input Training Parameters by User

parser = argparse.ArgumentParser(
    description="Train the UNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")
parser.add_argument("--mode", default="train", type=str, dest="mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

# Parsing args variable.
args = parser.parse_args()

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir
mode = args.mode
train_continue = args.train_continue

# make directories
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# print arguments
print("learning rate : %.4e" % lr)
print("batch size : %d" % batch_size)
print("number of epoch : %d" % num_epoch)
print("data dir : %s" % data_dir)
print("checkpoint dir : %s" % ckpt_dir)
print("log dir : %s" % log_dir)
print("result dir : %s" % result_dir)
print("mode : %s" % mode)
print("Train continue : %s" % train_continue)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.empty_cache()
else:
    device = torch.device('cpu')

## simple function for save img data.(tensor to numpy... etc)
fn_toNumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## make network instance
net = UNet().to(device)

## make optimizer and loss function
optim = torch.optim.Adam(params=net.parameters(), lr=lr)
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Preparing Dataset and Dataloader (train)
# num_workers : num of multiprocessor
# batch_size : the number of data in one batch unit

## Transforming data
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

## for Evaluation
dataset_eval = Dataset(data_dir=data_dir, dfSrc=None, transform=transform)
loader_eval = DataLoader(dataset=dataset_eval, batch_size=batch_size, shuffle=False, num_workers=8)
num_data_eval = len(dataset_eval)
num_batch_eval = np.ceil(num_data_eval / batch_size)

##
# Evaluation Process --------------------------------
# No back propagation step, only forward network.

## load network and optimizer
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
    net.eval()
    loss_arr = []

    for batch, data in enumerate(loader_eval, 1):
        # forward pass

        input = data['input'].to(device)

        # getting output by feeding input data.
        output = net(input)

        # set save data
        input = fn_toNumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_toNumpy(fn_class(output))

        print("TEST : BATCH %04d / %04d " % (batch, num_batch_eval))

        # save data samples..as png file
        if not os.path.exists(os.path.join(result_dir, 'png', 'test')):
            os.makedirs(os.path.join(result_dir, 'png', 'test'))

        # save data samples..as npy file
        plt.imsave(os.path.join(result_dir, 'png', 'test', 'TestB%04d_input.png' % batch), input[0].squeeze(), cmap='gray')
        plt.imsave(os.path.join(result_dir, 'png', 'test', 'TestB%04d_output.png' % batch), output[0].squeeze(), cmap='gray')
        np.save(os.path.join(result_dir, 'numpy', 'TestB%04d_input.npy' % batch), input[0].squeeze())
        np.save(os.path.join(result_dir, 'numpy', 'TestB%04d_output.npy' % batch), output[0].squeeze())
