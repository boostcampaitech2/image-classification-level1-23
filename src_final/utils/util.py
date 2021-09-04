import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

# mixup
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''
    mixup은 training에서만 사용한다.
    train에서는 정확도가 낮게나오지만(이미지는 두개지만 정답은 max값 1개로 취급하기 때문) test에서는 높게 나온다.
    이것도 label smoothing과 같은 선상에서 이해하면 좋을 것 같다.
    label smoothing은 incoding된 hard한 정답이 일반화에 방해가 되어 overfitting을 일으킨다는 가정에서 나온 기법인데
    mixup 또한 두개의 의미지를 섞어서 일반화를 위해 사용되는 것 같다.
    '''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()  # batch_size만큼의 random index 1차원 벡터\

    else:
        index = torch.randperm(batch_size)
    # pdb.set_trace()
    mixed_x = lam * x + (1 - lam) * x[index, :]  # 원본 batch image와 섞인 batch image들을 mix
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def restore(inp, title=None):
    """Imshow for Tensor."""
    # pdb.set_trace()
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
def plot_examples(images, title=None):
    fig = plt.figure(figsize=(10, 10))
    columns = 4
    rows = 4

    for i in range(1, len(images)+1):
        img = images[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        if title is not None:
            plt.title(title[i-1])
        plt.tight_layout()
    plt.show()

def mixs_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (18-lam) * criterion(pred, y_b)

def rand_bbox(size, lam): # size : [Batch_size, Channel, Width, Height]
    W = size[2] 
    H = size[3] 
    cut_rat = np.sqrt(1. - lam)  # 패치 크기 비율
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)  

   	# 패치의 중앙 좌표 값 cx, cy
    cx = np.random.randint(W)
    cy = np.random.randint(H)
		
    # 패치 모서리 좌표 값 
    bbx1 = 0
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = W
    bby2 = np.clip(cy + cut_h // 2, 0, H)
   
    return bbx1, bby1, bbx2, bby2

def cutmix_data(data, target, beta, device,):
    lam = np.random.beta(beta, beta)
    rand_index = torch.randperm(data.size()[0]).to(device)
    target_a = target # 원본 이미지 label
    target_b = target[rand_index] # 패치 이미지 label       
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    return data, target_a, target_b, lam