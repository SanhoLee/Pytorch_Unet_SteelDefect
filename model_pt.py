##
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

UNet = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet', classes=4, activation=None)