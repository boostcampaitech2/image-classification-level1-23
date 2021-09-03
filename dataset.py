# import python default libraries
import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List

# import pandas and numpy
import pandas as pd
import numpy as np

# import image processing related libraries
import cv2
from PIL import Image

# torch and torch vision
import torch
from torch.utils.data import Dataset, Subset, random_split
from torchvision import transforms
from torchvision.transforms import *


# import Albumentations for image / tensor transformations
import albumentations
from albumentations import *
from albumentations.pytorch import ToTensorV2


class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, masks, genders, ages, transform, device):
        """Initialize CustomDataset
        
        Parameters:
        img_paths (list of string): list of image paths
        labels (list of int): list of labels
        
        """
        self.img_paths = img_paths
        self.labels = torch.tensor(labels).to(device)
        self.masks = torch.tensor(masks).to(device)
        self.genders = torch.tensor(genders).to(device)
        self.ages = torch.tensor(ages).to(device)
        self.device = device
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        
        if self.transform:
            image = self.transform(image=np.array(image))['image']

        return image.to(self.device), (self.labels[index], self.masks[index], self.genders[index], self.ages[index])

    def __len__(self):
        return len(self.img_paths)

def get_transforms(mode =("train", "val"), mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)):
    """
    train 혹은 validation의 augmentation 함수를 정의합니다. train은 데이터에 많은 변형을 주어야하지만, validation에는 최소한의 전처리만 주어져야합니다.
    
    Args:
        mode: 'train', 혹은 'val' 혹은 둘 다에 대한 augmentation 함수를 얻을 건지에 대한 옵션입니다.
        img_size: Augmentation 이후 얻을 이미지 사이즈입니다.
        mean: 이미지를 Normalize할 때 사용될 RGB 평균값입니다.
        std: 이미지를 Normalize할 때 사용될 RGB 표준편차입니다.

    Returns:
        transformations: Augmentation 함수들.

    """
    transform_train = albumentations.Compose([
                #Resize(img_size[0], img_size[1], p=1.0),
                #Resize(200, 260, p=1.0),
                CenterCrop(height = 400, width = 200), # add centercrop 350/350 -> 400/200 -> 300/300
                #HorizontalFlip(p=0.5),
                #ShiftScaleRotate(p=0.5),
                #HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                #GaussNoise(p=0.5),
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)

    transform_val = albumentations.Compose([
                #Resize(img_size[0], img_size[1]),
                #Resize(200, 260),
                CenterCrop(height = 400, width = 200), # add centercrop
                Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.0)

    if mode == "train":
        return transform_train
    elif mode == "val":
        return transform_val


### 마스크 여부, 성별, 나이를 mapping할 클래스를 생성합니다.

class MaskLabels:
    mask = 0
    incorrect = 1
    normal = 2

class GenderLabels:
    male = 0
    female = 1

class AgeGroup:
    map_label = lambda x: 0 if int(x) < 30 else 1 if int(x) < 60 else 2
