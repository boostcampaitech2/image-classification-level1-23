import argparse
from gc import collect
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
## 
import transform.transform  as module_transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

#
from collections import Counter

class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
        self.transform = transforms.Compose([
                                                transforms.CenterCrop((400, 200)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.245))
                                            ])

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)

def main(config):
    logger = config.get_logger('test')

    test_dir = '/opt/ml/input/data/eval'
    
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id) for img_id in submission.ImageID]
    dataset = TestDataset(image_paths)

    loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=4
    )
    transform = config.init_obj("set_transform", module_transform) 

    # build model architecture
    model = config.init_obj('module', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    # loss_fn = getattr(module_loss, config['loss'])
    # metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    import sys
    pth = config["test"]["path"]
    checkpoint = torch.load(pth)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_predictions = []
    # for images in tqdm(loader):
    #     count =1 
    #     colloct_pred = []
    #     for transformer in transform.transformations["eavl"]:
    #         # print("-----" * 10, transformer, count, "-"*20)
    #         with torch.no_grad():
    #             augmented_image = transformer.augment_image(images)
    #             images = images.to(device)
    #             pred = model(images)
    #             pred = pred.argmax(1).item()
    #             colloct_pred.append(pred)
    #             count +=1
    #     pred = Counter(colloct_pred).most_common()[0][0]
    #     all_predictions.append(pred)
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred = model(images)
            pred = pred.argmax(1).item()
            all_predictions.append(pred)
    submission['ans'] = all_predictions
    submission.to_csv(os.path.join(test_dir, f"{config['test_name']}_submission.csv"), index=False)
    print('test inference is done!')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
