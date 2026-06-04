# dataset.py
import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FER2013Dataset(Dataset):
    """FER2013 自定义数据集类"""

    EMOTION_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    def __init__(self, csv_path, split='train', transform=None):
        """
            transform: torchvision 变换
        """
        self.transform = transform
        df = pd.read_csv(csv_path)

        # 按 Usage 列划分数据集
        if split == 'train':
            self.data = df[df['Usage'] == 'Training']
        elif split == 'val':
            self.data = df[df['Usage'] == 'PublicTest']
        elif split == 'test':
            self.data = df[df['Usage'] == 'PrivateTest']
        else:
            raise ValueError(f"Unknown split: {split}")

        self.data = self.data.reset_index(drop=True)

        # 将像素字符串转为 numpy 数组
        self.images = []
        self.labels = []
        for idx in range(len(self.data)):
            pixels = np.array(self.data.loc[idx, 'pixels'].split(), dtype=np.float32)
            self.images.append(pixels.reshape(48, 48))
            self.labels.append(int(self.data.loc[idx, 'emotion']))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # shape: (48, 48)
        label = self.labels[idx]

        # 转为 PIL Image
        image = Image.fromarray(image.astype(np.uint8), mode='L')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(augment=False):
    """
    获取数据变换
    Args:
        augment: 是否使用数据增强（训练集 True，验证/测试集 False）
    """
    if augment:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(48, scale=(0.9, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


def get_dataloaders(csv_path, batch_size=64, num_workers=2):
    """创建训练/验证/测试 DataLoader"""
    train_dataset = FER2013Dataset(csv_path, split='train', transform=get_transforms(augment=True))
    val_dataset = FER2013Dataset(csv_path, split='val', transform=get_transforms(augment=False))
    test_dataset = FER2013Dataset(csv_path, split='test', transform=get_transforms(augment=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader