# EncodedPixels(str) 을 이용해서, np 어레이 형태로 데이터를 불러올수 있게 준비한다.

##
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import cv2
from PIL import Image

from zipfile import ZipFile

import torch
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader

from albumentations import (HorizontalFlip, Normalize, Compose)
from albumentations.pytorch import ToTensorV2

data_dir = './datasets'
result_dir = './result'


## Dataloader

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, dfSrc=None, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), phase='train'):
        '''

        :param data_dir:
        :param dfSrc: pivot dataframe, { index : ImageID, columns : ClassId, values : Encoded_Pixels }
        :param transform:
        '''
        self.data_dir = data_dir
        self.dfSrc = dfSrc
        # self.transform = transform
        self.transform = get_transform(phase, mean, std)

        if self.dfSrc is not None:
            self.imgs_dir = os.path.join(data_dir, 'train_images')
            self.lst_input = self.dfSrc.index
        else:
            self.imgs_dir = os.path.join(data_dir, 'test_images')
            self.lst_input = os.listdir(self.imgs_dir)

        self.imgShape = np.asarray(
            Image.open(os.path.join(self.imgs_dir, self.lst_input[0]))).shape  # tuple (height, width, channels)

    #  Data length, just the size of all data in dataframe
    def __len__(self):
        return len(self.dfSrc) if self.dfSrc is not None else len(self.lst_input)

    #  Get a specific index data, method of // instanceName(index)
    def __getitem__(self, index):
        # make input Image
        input_PIL = Image.open(os.path.join(self.imgs_dir, self.dfSrc.iloc[index].name))
        input_ = np.asarray(input_PIL)

        # make label image
        if self.dfSrc is not None:
            # get label img(h,w,c=4) and imgId
            imgId, label_ = getLabelImg(dfSrc=self.dfSrc, idx=index, shape=input_.shape)

        else:
            # just copy input data into label, in order to avoiding error.
            label_ = input_

        # Check Dimension and add one dimension.
        if input_.ndim == 2:
            input_ = input_[:, :, np.newaxis]
        if label_.ndim == 2:
            label_ = label_[:, :, np.newaxis]

        # make data object as dict type for label, input
        label_ = label_.transpose((2, 0, 1))
        data = self.transform(image=input_, mask=label_)

        return data

##

##
def get_transform(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend([
            HorizontalFlip(p=0.5)
        ])

    list_transforms.extend([
        Normalize(mean=mean, std=std, p=1),
        ToTensorV2()
    ])

    list_transforms = Compose(list_transforms)
    return list_transforms


# imgShape.shape : (height, width, )
def getLabelImg(dfSrc, idx, shape):
    '''

    :param dfSrc: pivot dataframe.
    :param idx: row index of dataframe
    :param shape: image shape (h, w, c=useless)
    :return: fname, label(h,w,c=4)

    '''

    fname = dfSrc.iloc[idx].name
    labels = dfSrc.iloc[idx][:4]  # data of channels : 1~4
    masks = np.zeros((shape[0], shape[1], 4), dtype=np.float32)

    # labels.values : [1,2,3,4]
    for i, label in enumerate(labels.values):  # labels.values : list of each channel data
        if label is not np.nan:
            label = label.split(' ')
            position = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
            for pos, le in zip(position, length):
                mask[pos:(pos + le)] = 1

            # map to origianl masks variable.
            masks[:, :, i] = mask.reshape(shape[0], shape[1], order='F')

    return fname, masks


def filterDF(df, imgListInDir):
    '''

    :param df: raw dataframe(columns : ImageId, ClassId, EncodedPixels)
    :param imgListInDir: images in train_images directory.
    :return: filtered dataframe by images in the directory.
    '''

    # design df_filtered just like df, same columns
    columns = df.columns.values

    # for imgId in imgListInDir:
    for ImageId in list(df.ImageId):

        idx = df.loc[df['ImageId'] == ImageId].index[0]
        if ImageId not in imgListInDir:
            # drop this row.
            df = df.drop([idx], axis=0)

    return df

##
