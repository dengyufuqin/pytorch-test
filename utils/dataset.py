import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset
import random
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, datatxt, transform=None, target_transform=None):
        super(CustomDataset, self).__init__()
        imgs = []
        file_txt = open(datatxt, 'r')
        for line in file_txt:
            line = line.rstrip()
            words = line.split('|')
            imgs.append((words[0], words[1]))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        random.shuffle(self.imgs)
        name, label = self.imgs[index]
        img = Image.open(name).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = int(label)

        return img, label

    def __len__(self):
        return len(self.imgs)