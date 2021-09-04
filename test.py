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

def func_labels(outputs_mask, outputs_gender, outputs_age, device):
    """mask, gender, age의 조합을 통해 최종 18개의 class 중 하나를 반환한다.

    Args:
        outputs_mask: model의 forward 결과물중 mask에 대한 output
        outputs_gender: model의 forward 결과물중 gender에 대한 output
        outputs_age: model의 forward 결과물중 age에 대한 output 
        device: torch.device('cuda') 혹은 torch.device('cpu')

    Returns:
        outputs_label.to(device): ex) torch.Tensor([0, 17, 1, 3, ...]).to(device)
    """
    outputs_label = torch.Tensor([])
    len_outputs = len(outputs_mask)
    for i in range(len_outputs):
        mask_class = outputs_mask[i]
        _, mask_class = mask_class.max(dim=0)
        
        gender_class = outputs_gender[i]
        _, gender_class = gender_class.max(dim=0)
        
        age_class = outputs_age[i]
        _, age_class = age_class.max(dim=0)
        
        label = mask_class * 6 + gender_class * 3 + age_class
        
        #label: int -> [[1, 0, 0, 0]]
        one_hot = torch.zeros((1,18))
        one_hot[0][label] = 1
        label = one_hot
        outputs_label = torch.cat([outputs_label, label])
    return outputs_label.to(device)

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
    # state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    all_predictions = []
    for images in tqdm(loader):
        with torch.no_grad():
            images = images.to(device)
            pred_outputs_mask, pred_outputs_gender, pred_outputs_age = model(images)
            pred = func_labels(pred_outputs_mask, pred_outputs_gender, pred_outputs_age, device)
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
