# solution : OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# soution end 
# from torchvision import transforms, utilsㅁs
from .randAugment import RandAugment
import ttach as tta
import albumentations
from albumentations import *
from albumentations.pytorch import ToTensorV2
# set_tuple = lambda x : tuple([float(z) for z in x.split(",")])

class NO_resize_Base_C348TransForm(object):
    def __init__(self, mean, std, resize, use_rand_aug = False):
        self.mean = mean
        self.std = std
        self.x, self.y = resize # notuse    
        self.use_rand_aug = use_rand_aug
        self.get_transforms()

    def get_transforms(self,need=('train', 'val', "eavl")):
        self.transformations = {}
        if 'train' in need:
            self.transformations['train'] = transforms.Compose([
                                                transforms.CenterCrop((348,348)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=self.mean, std=self.std),
                                            ])
            if self.use_rand_aug:
                self.transformations["train"].transforms.insert(1, RandAugment())
        if 'val' in need:
            self.transformations['val'] = transforms.Compose([
                                                transforms.CenterCrop((348,348)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=self.mean, std=self.std),
                                            ])
        if 'eavl' in need:
            self.transformations['eavl'] = tta.Compose([
                                                tta.HorizontalFlip(),
                                                # tta.Rotate90(angles=[0, 90]),
                                                # tta.Scale(scales=[1, 2]),
                                                tta.FiveCrops(320, 160),
                                                # tta.Multiply(factors=[0.7, 1]),
                                            ])
        return self.transformations

class NO_resize_Base_C400_200_TransForm(object):
    def __init__(self, mean, std, resize, use_rand_aug = False):
        self.mean = mean
        self.std = std
        self.x, self.y = resize # notuse    
        self.use_rand_aug = use_rand_aug
        self.get_transforms()

    def get_transforms(self,need=('train', 'val', "eavl")):
        self.transformations = {}
        if 'train' in need:
            self.transformations['train'] = transforms.Compose([
                                                transforms.CenterCrop((400,200)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=self.mean, std=self.std),
                                            ])
            if self.use_rand_aug:
                self.transformations["train"].transforms.insert(1, RandAugment())
        if 'val' in need:
            self.transformations['val'] = transforms.Compose([
                                                transforms.CenterCrop((400,200)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=self.mean, std=self.std),
                                            ])
        if 'eavl' in need:
            self.transformations['eavl'] = tta.Compose([
                                                tta.HorizontalFlip(),
                                                # tta.Rotate90(angles=[0, 90]),
                                                # tta.Scale(scales=[1, 2]),
                                                tta.FiveCrops(320, 160),
                                                # tta.Multiply(factors=[0.7, 1]),
                                            ])
        return self.transformations

class CustomTransForm(object):
    def __init__(self, mean, std, resize, use_rand_aug = False):
        self.mean = mean
        self.std = std
        self.x, self.y = resize # notuse
        self.use_rand_aug = use_rand_aug
        self.get_transforms()
    def get_transforms(self,need=('train', 'val', "eavl")):
        self.transformations = {}
        if 'train' in need:
            self.transformations['train'] = albumentations.Compose([
                                                CenterCrop(height = self.x, width = self.y), # add centercrop 350/350 -> 400/200 -> 300/300
                                                RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                                                Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
                                                ToTensorV2(p=1.0),
                                            ], p=1.0)
            if self.use_rand_aug:
                self.transformations["train"].transforms.insert(1, RandAugment())
        if 'val' in need:
            self.transformations['val'] = albumentations.Compose([
                                                CenterCrop(height = self.x, width = self.y), # add centercrop 350/350 -> 400/200 -> 300/300
                                                Normalize(mean=self.mean, std=self.std, max_pixel_value=255.0, p=1.0),
                                                ToTensorV2(p=1.0),
                                            ], p=1.0)