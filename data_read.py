##
import os
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "./datasets"
origin_imgs_dir = os.path.join(data_dir, 'origin', 'train_images')

dir_save_train = os.path.join(data_dir, 'train')
dir_save_valid = os.path.join(data_dir, 'valid')
dir_save_test = os.path.join(data_dir, 'test')
df = pd.read_csv(os.path.join(data_dir, 'train.csv'))

##

# if not os.path.exists(dir_save_train):
#     os.makedirs(dir_save_train)
# if not os.path.exists(dir_save_valid):
#     os.makedirs(dir_save_valid)
# if not os.path.exists(dir_save_test):
#     os.makedirs(dir_save_test)

total_number_imgs = len(os.listdir(origin_imgs_dir))

ratio_valid_n_test = 0.2
ratio_train = 1 - ratio_valid_n_test

## set each dataset. Train, Valid and Test potion.
num_train_img = int(total_number_imgs * ratio_train)
num_valid_img = int((total_number_imgs - num_train_img) / 2)
num_test_img = total_number_imgs - (num_train_img + num_valid_img)
num_ratios = [num_train_img, num_valid_img, num_test_img]

# I think no need to files separate, just managing the number of its portion.


##test
imgId = df.ImageId[1000]
imgs_dir = os.path.join(data_dir,'origin', 'train_images')

input_PIL = Image.open(os.path.join(imgs_dir, imgId))
input = np.asarray(input_PIL)


label_en_Px = list(df[df['ImageId'] == imgId].EncodedPixels)[0].split(' ')
Px_Pos = map(int, label_en_Px[0::2])
Px_len = map(int, label_en_Px[1::2])
label_oneD = np.zeros(input.shape[0]*input.shape[1], dtype=np.uint8)

for pos, len in zip(Px_Pos, Px_len):
    label_oneD[pos:pos+len-1] = 255

label = label_oneD.reshape(input.shape[0], input.shape[1], order='F')

if label.ndim == 2:
    label = label[ : , : , np.newaxis]
if input.ndim == 2:
    input = input[ : , : , np.newaxis]

    
    
##
plt.subplot(211)
plt.imshow(input, cmap='gray')

plt.subplot(212)
plt.imshow(label, cmap='gray')

plt.title(imgId)
plt.show()
##

