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

data_dir = './datasets'


## Dataloader

class Dataset(torch.utils.data.Dataset):
    # def __init__(self, data_dir, transform=None):
    def __init__(self, data_dir, transform=None):
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

        # data의 transform이 있을 경우에는 데이터 적용한다
        self.data_dir = data_dir
        self.imgs_dir = os.path.join(data_dir, 'train_images')
        self.transform = transform
        self.df = pd.read_csv(os.path.join(data_dir, 'train.csv'))  # for label data...

        # input list variable, in order to avoid not existing element in the list,
        # Specifying the input list based on csv file.
        lst_input = list(self.df.ImageId)
        self.lst_input = lst_input
        self.imgShape = np.asarray(Image.open(os.path.join(self.imgs_dir, self.lst_input[0]))).shape  # tuple (height, width, channels)


    # __init__ 안에다가 함수를 정의하면, init 안에서만 call이 가능하다
    def getLabelImg(self, imgId):
        label_en_Px = list(self.df[self.df['ImageId'] == imgId].EncodedPixels)[0].split(' ')
        Px_Pos = map(int, label_en_Px[0::2])
        Px_len = map(int, label_en_Px[1::2])
        label_oneD = np.zeros(self.imgShape[0] * self.imgShape[1], dtype=np.uint8)

        for pos, len in zip(Px_Pos, Px_len):
            label_oneD[pos:pos + len - 1] = 255

        label = label_oneD.reshape(self.imgShape[0], self.imgShape[1], order='F')

        return label

    #  Data length, just the size of all data in dataframe
    def __len__(self):
        return len(self.lst_input)

    #  Get a specific index data, method of // instanceName(index)
    def __getitem__(self, index):
        imgId = self.lst_input[index]

        # make input Image
        input_PIL = Image.open(os.path.join(self.imgs_dir, imgId))
        input = np.asarray(input_PIL)

        # make label image
        label = self.getLabelImg(imgId)

        # make array value into 0 to 1.
        input = input / 255.0
        label = label / 255.0

        # Check Dimenstion and add one dimension.
        if input.ndim == 2:
            input = input[:, :, np.newaxis]
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        # make data object as dict type for label, input
        data = {'input': input, 'label': label}
        # data = {'input': input}

        # if self.transform:
        #     data = self.transform(data)
        #     # transform 클래스의 return 값은 여기서 선언한 data 사전형과 동일하게 해줘야 한다.

        return data


## test dataset
dataset_train = Dataset(data_dir='datasets')

##
data = dataset_train.__getitem__(10)
input = data['input']
label = data['label']

# plot data
ax1 = plt.subplot(211)
ax1.set_title('input')
plt.imshow(input, cmap='gray')

ax2 = plt.subplot(212)
plt.imshow(label, cmap='gray')
ax2.set_title('label')

plt.show()

##

