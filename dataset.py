# EncodedPixels(str) 을 이용해서, np 어레이 형태로 데이터를 불러올수 있게 준비한다.

##
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from zipfile import ZipFile

import torch
import torch.utils.data
from torchvision import transforms
from torch.utils.data import DataLoader

data_dir = './datasets'


## Dataloader

class Dataset(torch.utils.data.Dataset):
    # def __init__(self, data_dir, transform=None):
    def __init__(self, data_dir, dfSrc=None, transform=None):
        '''
        >>
            origin data :
                1. input img filename,
                2. label encoded string,
                3. defect class id ?
            convert to  :
                1. input img    (numpy array),
                2. label img    (numpy array),
                3. defect class id ?
        '''

        def filterDF(df, imgListInDir):
            '''

            :param df: raw dataframe(columns : ImageId, ClassId, EncodedPixels)
            :param imgListInDir: images in train_images directory.
            :return: filtered dataframe by images in the directory.
            '''

            # design df_filtered just like df, same columns
            columns = ['ImageId', 'EncodedPixels']
            df_filtered = pd.DataFrame(columns=columns)

            i = 0

            for imgId in imgListInDir:

                # searching imgId is exists in dataframe.
                t1 = np.where(df['ImageId'] == imgId)[0]

                # 디렉토리 파일이 df 에 없는 경우, continue
                if len(t1) == 0:
                    continue

                else:
                    # 디렉토리 파일이 df 에 있는 경우, df_filtered 추가
                    selected_indices = list(t1)
                    df_filtered = pd.concat([df_filtered, df.iloc[selected_indices]], axis=0)

            return df_filtered

        # data의 transform이 있을 경우에는 데이터 적용한다
        self.data_dir = data_dir
        self.dfSrc = dfSrc
        self.transform = transform



        if self.dfSrc is not None:
            self.imgs_dir = os.path.join(data_dir, 'train_images')

            # filtering with img List in train directory...
            self.dfSrc = filterDF(dfSrc, list(os.listdir(self.imgs_dir)))
            self.lst_input = list(self.dfSrc.ImageId)
        else:
            self.imgs_dir = os.path.join(data_dir, 'test_images')
            self.lst_input = os.listdir(self.imgs_dir)

        self.imgShape = np.asarray(
            Image.open(os.path.join(self.imgs_dir, self.lst_input[0]))).shape  # tuple (height, width, channels)

    # __init__ 안에다가 함수를 정의하면, init 안에서만 call이 가능하다
    def getLabelImg(self, imgId):
        label_en_Px = list(self.dfSrc[self.dfSrc['ImageId'] == imgId].EncodedPixels)[0].split(' ')
        Px_Pos = map(int, label_en_Px[0::2])
        Px_len = map(int, label_en_Px[1::2])
        label_oneD = np.zeros((self.imgShape[0] * self.imgShape[1]), dtype=np.uint8)

        for pos, leng in zip(Px_Pos, Px_len):
            label_oneD[pos:pos + leng - 1] = 255

        label = label_oneD.reshape((self.imgShape[0], self.imgShape[1], 1), order='F')

        return label

    #  Data length, just the size of all data in dataframe
    def __len__(self):
        return self.dfSrc.shape[0] if self.dfSrc is not None else len(self.lst_input)

    #  Get a specific index data, method of // instanceName(index)
    def __getitem__(self, index):
        imgId = self.lst_input[index]

        # make input Image
        input_PIL = Image.open(os.path.join(self.imgs_dir, imgId))
        input = np.asarray(input_PIL)

        # squeeze input data into 1 channel.
        input = np.delete(input, [1, 2], axis=2)

        # make label image
        if self.dfSrc is not None:
            # Get label img and make img pixel value into 0 to 1.
            label = self.getLabelImg(imgId)
        else:
            # just copy input data into label, in order to avoid error.
            label = input

        input = input / 255.0
        label = label / 255.0

        # Check Dimension and add one dimension.
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        # make data object as dict type for label, input
        data = {'input': input, 'label': label}
        # data = {'input': input}

        if self.transform:  # only when training model.
            data = self.transform(data)
        # transform 클래스의 return 값은 여기서 선언한 data 사전형과 동일하게 해줘야 한다.

        return data


##
class ToTensor(object):
    '''
    change data type, numpy -> tensor
    '''

    def __call__(self, data):
        # get data
        label, input = data['label'], data['input']

        # set each column into tensor-way// (width, height, channel) -> (channel, width, height)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data


class Normalization(object):
    '''
    Normalizing pixel values
    '''

    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        # only to input data
        label, input = data['label'], data['input']
        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}
        return data


class RandomFlip(object):
    '''
    Flip data left and right, Up and Down Randomly
    '''

    def __call__(self, data):
        # get data
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}
        return data

## test dataset
# data = Dataset(data_dir=data_dir).__getitem__(10)
# input = data['input']
# label = data['label']
#
# ##
# print("Input SHAPE : ", input.shape)
# print('type of input data : %s' % type(input))
# print('type of label data : %s' % type(label))
#
# ## plot data
# ax1 = plt.subplot(211)
# ax1.set_title('input')
# plt.imshow(input, cmap='gray')
#
# ax2 = plt.subplot(212)
# plt.imshow(label, cmap='gray')
# ax2.set_title('label')
#
# plt.show()

##
