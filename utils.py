##
import os
import pandas as pd
import numpy as np
import torch


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

    i = 0

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


def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict()
    },
        # "%s/model_epoch%d.pth" % (ckpt_dir, epoch)
        "%s/model_epoch.pth" % (ckpt_dir)
    )


def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    # ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    # epoch = int(ckpt_lst[-1].split('epoch')[-1].split('.pth')[0])
    epoch = 0

    return net, optim, epoch
