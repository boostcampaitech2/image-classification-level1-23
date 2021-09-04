# from torchvision import transforms
from base import BaseDataLoader
# from torch.utils.data import DataLoader


class BaseLoader(BaseDataLoader): # 
    def __init__(self, data_set, batch_size, shuffle=True, validation_split=0.0, num_workers=1,): # TODO: , training=True !!! what is this
        self.data_set = data_set
        super().__init__(self.data_set, batch_size, shuffle, validation_split, num_workers)
        pass




# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#         ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        # super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)