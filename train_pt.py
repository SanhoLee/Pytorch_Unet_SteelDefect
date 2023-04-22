## necessary modules
import argparse
import os
import numpy as np
import pandas as pd

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from dataset import *
from utils import *
from model_pt import UNet

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

## summaryWriter
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, "train"))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, "val"))

## simple function for save img data.(tensor to numpy... etc)
fn_toNumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: (x > 0.2) * 1

#
## make network instance
net = UNet.to(device)

## make optimizer and loss function
optim = torch.optim.Adam(params=net.parameters(), lr=lr)
fn_loss = nn.BCEWithLogitsLoss().to(device)

##schedular
schedular = ReduceLROnPlateau(optimizer=optim, mode='min', patience=2, factor=0.3, verbose=True)

## Preparing Dataset and Dataloader (train)
# num_workers : num of multiprocessor
# batch_size : the number of data in one batch unit.

# Reading DF from CSV file
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

## split DF into 2 sets, Train(0.8) and Valid(0.2) / All data is type of Dataframe.
df = filterDF(df=df, imgListInDir=os.listdir(os.path.join(data_dir, 'train_images')))
df = df.pivot(index='ImageId', columns='ClassId',
              values='EncodedPixels')  # pivot shape 으로 변형해서, 한 이미지 아이디에 대해서, 클래스별 'EncodedPixels' 값을 할당해준다.
df['defects'] = df.count(axis=1)  # column direction.

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['defects'], random_state=69)

print("The Number of TRAIN Data : %d " % len(train_df))
print("The Number of VALID Data : %d " % len(val_df))

## Transforming data

# get this value from pre-trained model of resnet18 in the repository.
mean_in = (0.485, 0.456, 0.406)
std_in = (0.229, 0.224, 0.225)

# set transform object.
# transform = transforms.Compose([Normalization(mean=mean_in, std=std_in), RandomFlip(), ToTensor()])

## for train
dataset_train = Dataset(data_dir=data_dir, dfSrc=train_df, mean=mean_in, std=std_in, phase='train')
loader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
num_data_train = len(dataset_train)
num_batch_train = np.ceil(num_data_train / batch_size)  # 전체 데이터에 대해 train 을 반복되는 iteration 횟수와 같은 개념이 된다.

## for validation
dataset_val = Dataset(data_dir=data_dir, dfSrc=val_df, mean=mean_in, std=std_in, phase='valid')
loader_val = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)
num_data_val = len(dataset_val)
num_batch_val = np.ceil(num_data_val / batch_size)

## start train process...
st_epoch = 0

# TRAIN mode
if (train_continue == 'on'):
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

for epoch in range(st_epoch + 1, num_epoch + 1):
    # set train model explicitly
    net.train()
    loss_arr = []

    # train process, initialize gradient of tensors for each epoch.
    optim.zero_grad()

    for batch, data in enumerate(loader_train, 1):

        # forward pass
        label = data['mask'].to(device)
        input = data['image'].to(device)
        output = net(input)
        # output = torch.sigmoid(output)

        # backward pass
        loss = fn_loss(output, label)
        loss.backward()

        # update gradients
        optim.step()
        optim.zero_grad()   # in order to initializing previous gradient value by each mini-batch number.

        # save loss value
        loss_arr += [loss.item()]

        print("TRAIN : EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" % (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

    # save network and numpy file at specified checkpoint.
    if epoch % 5 == 0:
        # set save data
        input = fn_toNumpy(input)
        label = fn_toNumpy(label)
        # output = fn_toNumpy(fn_class(output))

        # save(ckpt_dir, net, optim, epoch)
        np.save(os.path.join(result_dir, 'numpy', 'epoch%04d_input.npy' % epoch), input)
        np.save(os.path.join(result_dir, 'numpy', 'epoch%04d_label.npy' % epoch), label)
        np.save(os.path.join(result_dir, 'numpy', 'epoch%04d_output.npy' % epoch), output)

    # save loss value.
    writer_train.add_scalar("loss", np.mean(loss_arr), epoch)

    # Validation Process --------------------------------
    # No back propagation step, only forward network.

    with torch.no_grad():
        net.eval()
        loss_arr = []

        # train process, initialize gradient of tensors for each epoch.
        optim.zero_grad()

        for batch, data in enumerate(loader_val, 1):
            # forward pass
            label = data['mask'].to(device)
            input = data['image'].to(device)
            output = net(input)
            # output = torch.sigmoid(output)

            # skip backward propagation
            loss = fn_loss(output, label)

            # set schedular...
            schedular.step(loss)

            # save loss value
            loss_arr += [loss.item()]

            print("Valid : EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" % (
                epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

            # input을 denorm 하는 것은 꼭 필요한 과정은 아니므로, 생략하고 돌려본다.
            # input = fn_toNumpy(input)
            # label = fn_toNumpy(label)
            # output = fn_toNumpy(fn_class(output))

            # print('after valid ----- batch %04d ' % batch)
            # print('Shape of Label : ', label.shape)
            # print('Shape of Output : ', output.shape)
            #
            # print('When saving png and numpy file ----- epoch %04d ' % epoch)
            # print('Shape of Label : ', label.shape)
            # print('Shape of Output : ', output.shape)

        # save loss value for valid process.
        writer_val.add_scalar("loss", np.mean(loss_arr), epoch)

writer_train.close()
writer_val.close()
