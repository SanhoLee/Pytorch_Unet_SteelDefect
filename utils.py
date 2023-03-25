##
import os
import pandas as pd
import numpy as np



data_dir = './datasets'
imgs_dir = os.path.join(data_dir, 'train_images')
##
def filterDF(df, imgListInDir):
    '''

    :param df: raw dataframe(columns : ImageId, ClassId, EncodedPixels)
    :param imgListInDir: images in train_images directory.
    :return: filtered dataframe by images in the directory.
    '''

    # design df_filtered just like df, same columns
    columns = ['ImageId', 'ClassId', 'EncodedPixels']
    df_filtered = pd.DataFrame(columns=columns)

    i=0

    for imgId in imgListInDir:

        # searching imgId is exists in dataframe.
        t1 = np.where(df['ImageId'] == imgId)[0]

        # 디렉토리 파일이 df 에 없는 경우, continue
        if len(t1) == 0:
            continue

        else:
            # 디렉토리 파일이 df 에 있는 경우, df_filtered 추가
            # df_filtered = df_filtered.append([df[df['ImageId'] == imgId]], columns)
            selected_indices = list(t1)
            df_filtered = df_filtered.append(df.iloc[selected_indices], columns)

    return df_filtered


## filtering with img List in train directory...
#
# imgListInDir = list(os.listdir(imgs_dir))
# df = pd.read_csv(os.path.join(data_dir, 'train.csv'))  # for label data...
#
# ##
# df_filtered = filterDF(df, imgListInDir)
##

