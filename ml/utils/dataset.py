import torch
from torch.utils.data import Dataset

import cv2

import os


class CarDataset(Dataset):
    def __init__(self, df, path, transform=None):
        super(CarDataset, self).__init__()
        self.transform = transform
        self.df = df.copy()
        self.path = path

    def __getitem__(self, item):
        img_name, rot_angle, speed = self.df.iloc[item].values
        img = cv2.imread(os.path.join(self.path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        measurements = torch.FloatTensor([rot_angle, speed])
        return img, measurements

    def __len__(self):
        return len(self.df)
