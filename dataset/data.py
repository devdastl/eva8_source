#custom module to create dataset

import torch
from torchvision import datasets, transforms
import torch
from PIL import Image
import numpy as np
from transform_albumentaion import data_albumentation

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, train=True):
        train_transform, test_transform = data_albumentation(horizontalflip_prob=0.5,
                                                                 rotate_limit=15,
                                                                 shiftscalerotate_prob=0.25,
                                                                 num_holes=1,cutout_prob=0.5)
        self.cuda = torch.cuda.is_available()
        self.required_transform = train_transform if train else test_transform
        self.dataset = getattr(datasets, dataset_name)(root='./data',
                                                       download=True, train=train, transform=None)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.required_transform(image=np.array(image))["image"]
        
        return image, label

    def __len__(self):
        return len(self.dataset)


class DataLoader():
    def __init__(self, dataset_name, batch_size=64):
        self.dataset_train = Dataset(dataset_name, train=True)
        self.dataset_test = Dataset(dataset_name, train=False)
        self.batch_size = batch_size
        self.cuda = torch.cuda.is_available()


    def get_loader(self):
        dataloader_args = dict(shuffle=True, batch_size=self.batch_size, num_workers=4, pin_memory=True) if self.cuda else dict(shuffle=True, batch_size=64)
        train_dataloader = torch.utils.data.DataLoader(self.dataset_train, **dataloader_args)
        test_dataloader = torch.utils.data.DataLoader(self.dataset_test, **dataloader_args)  

        return train_dataloader, test_dataloader
