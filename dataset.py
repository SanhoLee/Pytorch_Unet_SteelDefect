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

        :param data_dir:
        :param dfSrc: pivot dataframe, { index : ImageID, columns : ClassId, values : Encoded_Pixels }
        :param transform:
        '''
        self.data_dir = data_dir
        self.dfSrc = dfSrc
        self.transform = transform

        if self.dfSrc is not None:
            self.imgs_dir = os.path.join(data_dir, 'train_images')
            self.lst_input = self.dfSrc.index
        else:
            self.imgs_dir = os.path.join(data_dir, 'test_images')
            self.lst_input = os.listdir(self.imgs_dir)

        self.imgShape = np.asarray(Image.open(os.path.join(self.imgs_dir, self.lst_input[0]))).shape  # tuple (height, width, channels)

    #  Data length, just the size of all data in dataframe
    def __len__(self):
        return len(self.dfSrc) if self.dfSrc is not None else len(self.lst_input)

    #  Get a specific index data, method of // instanceName(index)
    def __getitem__(self, index):
        # make input Image
        input_PIL = Image.open(os.path.join(self.imgs_dir, self.dfSrc.iloc[index].name))
        input = np.asarray(input_PIL)

        # make label image
        if self.dfSrc is not None:
            # get label img(h,w,c=4) and imgId
            imgId, label = getLabelImg(dfSrc=self.dfSrc, idx=index, shape=input.shape)
        else:
            # just copy input data into label, in order to avoiding error.
            label = input

        input = input / 255.0
        label = label / 255.0

        print('input shape : ', input.shape)
        print('label shape : ', label.shape)

        # Check Dimension and add one dimension.
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        # make data object as dict type for label, input
        data = {'input': input, 'label': label}

        if self.transform:  # only when training model.
            data = self.transform(data)
        # transform 클래스의 return 값은 여기서 선언한 data 사전형과 동일하게 해줘야 한다.

        return data


##

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
                mask[pos:pos + le - 1] = 255

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


# ## test code.......
# df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
#
# ##
# df = filterDF(df=df, imgListInDir=os.listdir(os.path.join(data_dir, 'train_images')))
#
# ##
# df = df.pivot(index='ImageId', columns='ClassId',
#               values='EncodedPixels')  # pivot shape 으로 변형해서, 한 이미지 아이디에 대해서, 클래스별 'EncodedPixels' 값을 할당해준다.
# df['defects'] = df.count(axis=1)  # column direction.
#
# ## Transforming data
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
# data = Dataset(data_dir=data_dir, dfSrc=df, transform=transform)
#
# ##
# data = data.__getitem__(10)
#
# ##
# input = data['input']
# label = data['label']
#
# ##
# print("Input SHAPE : ", input.shape)
# print("label SHAPE : ", label.shape)
#
# print('type of input data : ', type(input))
# print('type of label data : ', type(label))
#
# ## simple function for save img data.(tensor to numpy... etc)
# fn_toNumpy = lambda x: x.to('cpu').detach().numpy().transpose(1, 2, 0)
# fn_denorm = lambda x, mean, std: (x * std) + mean
# fn_class = lambda x: 1.0 * (x > 0.5)
#
# ##
# label = fn_toNumpy(label)
# input = fn_toNumpy(fn_denorm(input, mean=0.5, std=0.5))
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
