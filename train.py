## necessary modules
import argparse
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import matplotlib.pyplot as plt

from dataset import *
from model import UNet

## input Training Parameters by User

parser = argparse.ArgumentParser(
    description="Train the UNet",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=10, type=int, dest="batch_size")
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



## Preparing Dataset and Dataloader
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor])
dataset_train = Dataset(data_dir=data_dir, transform=transform)
print('--------complete dataset_train -------------')
# loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
print('--------complete loader_train -------------')

# num_workers : num of multi-processor
# batch_size : the number of data in one batch unit.

num_data_train = len(dataset_train)
num_batch_train = np.ceil(num_data_train / batch_size)   # 전체 데이터에 대해 train 을 반복되는 iteration 횟수와 같은 개념이 된다.

## make network instance
net = UNet().to(device)


## make optimizer and loss function
optim = torch.optim.Adam(params=net.parameters(), lr=lr)
fn_loss = nn.BCEWithLogitsLoss().to(device)
print('--------complete loss and optim component -------------')

## train model
net.train()
loss_arr = []

for batch, data in enumerate(loader_train, 1):


    # forward pass
    label = data['input'].to(device)
    input = data['input'].to(device)

    # getting output by feeding input data.
    output = net(input)

    # backward pass
    optim.zero_grad()

    # getting loss
    loss = fn_loss(output, label)
    loss.backward()

    optim.step()

    # save loss value
    loss_arr += [loss.item()]

    print("TRAIN : BATCH %04d / %04d | LOSS %.4f" % (batch, num_batch_train, np.mean(loss_arr)))






## some works for saving data : checkpoint for network, optimizer and epoch info.





## need to consider how to make output string of masked pixels.




