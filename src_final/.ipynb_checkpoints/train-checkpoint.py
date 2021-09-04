import argparse
import collections
import random
from pyexpat import model
import torch
import numpy as np
import model.metric as module_metric
from parse_config import ConfigParser, run_id
from trainer import Trainer
from trainer.trainer_multilabel import CostumTrain
import torch.utils.data as data
from utils import prepare_device, write_json
from tqdm import tqdm
import pandas as pd
import os
##
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
##
import data_loader.dataset as module_dataset
import data_loader.data_loaders as module_data
import transform.transform  as module_transform
import model.model as module_model

import model.loss as module_loss
import model.optimizer as module_optimizer
from adamp import AdamP
##
import gc
import sys
import glob

import matplotlib.pyplot as plt
# fix random seeds for reproducibility

# SEED = 123
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)  # if use multi-GPU
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# np.random.seed(SEED)
# random.seed(SEED)]

device = torch.device('cuda')
train_csv = pd.read_csv(os.path.join("/opt/ml/input/data/train", 'train.csv'))

# input data, output data 리스트 만들기(이미지 텐서화는 CustomDataset에서 이뤄짐)
train_image_paths = []
train_labels = []
train_masks = [] # mask:0 / incorrect:1 / notwear:2
train_genders = [] # male:0 / female:1
train_ages = [] # (,30):0 / [30, 58):1 / [58,):2 

dict_mask = {'mask1':0,
             'mask2':0,
             'mask3':0,
             'mask4':0,
             'mask5':0,
             'incorrect_mask':1,
             'normal':2,
            }

dict_gender = {'male':0,
              'female':1}

for i in range(train_csv.shape[0]): # number of train image folders is 2700
    row = train_csv.loc[i]
    seven_paths = glob.glob("/opt/ml/input/data/train" + '/images/' + row['path'] + '/*.*')
    
    gender = row['gender']
    age = row['age']
    for i, path in enumerate(seven_paths):
        label = 0
        mask = path.split('/')[-1].split('.')[0]
        mask_label = dict_mask[mask]
        gender_label = dict_gender[gender]
        age_label = 0
        if 30 <= age < 58:
            age_label += 1
        elif 58 <= age:
            age_label += 2
            
        label = mask_label * 6 + gender_label * 3 + age_label        
                    
        train_image_paths.append(path)
        train_labels.append(label)
        train_masks.append(mask_label)
        train_genders.append(gender_label)
        train_ages.append(age_label)


def func_eval(model,data_iter,device):
    with torch.no_grad():
        n_total,n_correct = 0,0
        model.eval() # evaluate (affects DropOut and BN)
        for batch_in,batch_out in data_iter:
            y_trgt = batch_out.to(device)
            model_pred = model(batch_in.to(device))
            _,y_pred = torch.max(model_pred.data,1)
            n_correct += (y_pred==y_trgt).sum().item()
            n_total += batch_in.size(0)
        try:
            val_accr = (n_correct/n_total)
        except ZeroDivisionError:
            val_accr = 0
        model.train() # back to train mode 
    return val_accr

def main(config):
    logger = config.get_logger('train')
    # setting your dataset 
    device, device_ids = prepare_device(config['n_gpu'])
    print(f"device {device}, device_ids {device_ids}")

    print(f"start data set\n")
#     print(train_ages)
#     data_set = config.init_obj("data_set", module_dataset, device=device,)
    # set_transform 

    transform = config.init_obj("set_transform", module_transform) 
    print(f"end set set_transform\n")

    # # setup data_loader instances
    if config["data_loader"]["type"] == "BaseLoader":
        data_loader = config.init_obj('data_loader', module_data, data_set=data_set)
        val_data_loader = data_loader.split_validation()
        data_loader.data_set.set_transform(transform.transformations["train"])
        val_data_loader.dataset.set_transform(transform.transformations["val"])
        data_loader.data_set.set_transform(transform.transformations["train"])
    else: # multi label
#         def __init__(self, dir_path, labels, masks, genders, ages, transform, device):
#         train_image_paths = os.path.join(config["data_set"]["args"]["dir_path"], "image/")
        data_set = module_dataset.CustomDataset(train_image_paths, train_labels, train_masks, train_genders, train_ages, transform.transformations["train"], device)
        # data_set.set_transform(transform.transformations["train"])
        pass
    print(len(data_set))
    print(f"end set data_loader\n")

    # build model architecture, then print to console
    model = config.init_obj('module', module_model)
    # prepare for (multi-device) GPU training
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    if config["set_loss"]["type"] == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = config.init_obj("set_loss", module_loss)
    
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())\
    # FIXME : 커스텀 가능하게 수정
    optimizer = AdamP(model.parameters(), lr=config["optimizer"]["args"]["lr"], betas=config["optimizer"]["args"]["betas"], weight_decay= config["optimizer"]["args"]["weight_decay"])
    # optimizer = config.init_obj('optimizer', optim, trainable_params)
    # FIXME : 커스텀 가능하게 수정
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["lr_scheduler"]["args"]["milestones"], gamma=config["lr_scheduler"]["args"]["gamma"])

    # print(f"func_eval test")
    # train_accr = func_eval(model, data_loader, device)
    # valid_accr = func_eval(model, val_data_loader, device)
    # print ("train_accr:[%.3f] valid_accr:[%.3f]."%(train_accr,valid_accr))

    torch.cuda.empty_cache()   
    gc.collect()
    

    # FIXME: stkfold, multi label, basic 모두 사용할 수 있게
    print("train start")
    print(config["trainer"]["type"])
    if config["trainer"]["type"] == "multi":
        trainer = CostumTrain(model = model, config = config, criterion = criterion, optimizer = optimizer,
                                device = device, data_set = data_set, transform= transform,scheduler =lr_scheduler)
        pass
    
    else:
        trainer = Trainer(model, criterion, metrics, optimizer,
                            config=config,
                            device=device,
                            # data_loader=data_loader,                  # use stkfold
                            # valid_data_loader=val_data_loader,
                            data_set = data_set,
                            transform = transform,
                            lr_scheduler=lr_scheduler,
                            mix_up=config["set_transform"]["mix_up"],
                            cut_mix=config["set_transform"]["cut_mix"],
                        )

                        
    trainer.train()
 
    del trainer
    # del data_loader
    # del val_data_loader
    gc.collect()
    # test_dir = '/opt/ml/input/data/eval'
    
    # image_dir = os.path.join(test_dir, 'images')
    # submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    # image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    # # print(image_paths)?
    # eavl_dataset = module_dataset.TestDataset(image_paths)
    # eavl_dataset.set_transform(transform.transformations["val"])
    # loader = DataLoader(
    #     eavl_dataset,
    #     shuffle=True,
    # )

    # model.eval()
    # all_predictions = []
    # print(f"start answer")
    # for images in tqdm(loader):
    #     with torch.no_grad():
    #         images = images.to(device)
    #         pred = model.forward(images)
    #         pred = pred.argmax(dim=-1)
    #         all_predictions.extend(pred.cpu().numpy())
    # submission['ans'] = all_predictions
    # from datetime import datetime
    # from utils import read_json, write_json
    # import json
    # # # 제출할 파일을 저장합니다.

    # submission.to_csv(os.path.join(test_dir, f'{run_id}_submission.csv'), index=False)
    
    print('test inference is done!')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
