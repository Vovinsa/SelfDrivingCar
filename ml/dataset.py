import torch
from torch.utils.data import Dataset

import cv2

import os


class CarDataset(Dataset):
    def __init__(self, df, transform=None):
        super(CarDataset, self).__init__()
        self.transform = transform
        self.df = df.copy()

    def __getitem__(self, item):
        img_name, rot_angle, command = self.df.iloc[item].values()
        img = cv2.imread(os.path.join(self.path, img_name))
        if self.transform:
            img = self.transform(img)
        rot_angle = torch.IntTensor(rot_angle)
        command = torch.IntTensor(command)
        return img, rot_angle, command

    def __len__(self):
        return len(self.df)
