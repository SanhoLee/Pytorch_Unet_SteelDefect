# EncodedPixels(str) 을 이용해서, np 어레이 형태로 데이터를 불러올수 있게 준비한다.

##
import os
import numpy as np
import pandas as pd
from PIL import Image

from zipfile import ZipFile

import torch
import torch.utils.data

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
        self.transform = transform
        self.df = pd.read_csv(os.path.join(data_dir, 'train.csv'))  # for label data...

        # input list variable
        lst_input = os.listdir(os.path.join(data_dir, 'origin'))
        self.lst_input = lst_input

        def getLabelImg(self, imgId):
            return 0





##  Data length, just the size of all data in dataframe
    def __len__(self):
        return len(self.lst_input)

    #  Get a specific index data, method of // instanceName(index)
    def __getitem__(self, index):

        imgId = self.lst_input[index]

        input_PIL = Image.open(os.path.join(self.data_dir, 'origin', imgId))
        input = np.asarray(input_PIL)

        label_en_Px = list(self.df[self.df['ImageId'] == imgId])[0].split(' ')
        Px_Pos = label_en_Px[0::2]
        Px_len = label_en_Px[1::2]
        label = np.zeros(input.shape)



        # np array(npy) 파일을 불러 오기
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))

        # input = np.load(os.path.join(self.data_dir, imgId))
        # load from zip file
        # with ZipFile()

        # input = np.load(os.path.join(self.data_dir, imgId))
        # label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        #
        # # make array value into 0 to 1.
        # label = label / 255.0
        # input = input / 255.0
        #
        # #      어레이 차원 확인 후, 3차원 매트릭스로 변경, channel 데이터 인덱스 자리를 만들기 위해서 ?
        # if label.ndim == 2:
        #     label = label[:, :, np.newaxis]
        # if input.ndim == 2:
        #     input = input[:, :, np.newaxis]

        #      label, input 데이터를 사전형으로 준비
        data = {'input': input, 'label': label}

        # if self.transform:
        #     data = self.transform(data)
        #     # transform 클래스의 return 값은 여기서 선언한 data 사전형과 동일하게 해줘야 한다.

        return data
