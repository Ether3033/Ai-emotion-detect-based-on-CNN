# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelA_Small(nn.Module):
    """
    小型 CNN：2 层卷积，5×5 大卷积核，快速捕获全局特征
    参数量约 450K
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # 卷积层1: 1→32, 5×5 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), #一张图片输入 输出32个信息通道，输出5*5特征图，填充2行列补图保证输出一致
            nn.BatchNorm2d(32), #归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48->24 最大池化
            nn.Dropout2d(0.25), #0.25丢弃 鲁棒性提升
        )
        # 卷积层2: 32→64, 5×5 conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2), #32信息通道变64信息通道，5*5特征图
            nn.BatchNorm2d(64), #归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24->12 最大池化
            nn.Dropout2d(0.25),
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 256), #线性变换乘权重矩阵计算
            nn.ReLU(inplace=True), #relu激活函数
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x): #前向传播 按顺序执行各个网络层，输出结果
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc(x)
        return x


class ModelB_Medium(nn.Module):
    """
    中型 CNN：3 层卷积，3×3 小卷积核堆叠（等效更大感受野）
    参数量约 1.5M
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # 卷积层1: 1→64, 3×3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), #归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48->24
            nn.Dropout2d(0.25), #0.25丢弃 鲁棒性提升
        )
        # 卷积层2: 64→128, 3×3 conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), #归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24->12
            nn.Dropout2d(0.25),
        )
        # 卷积层3: 128→256, 3×3 conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12->6
            nn.Dropout2d(0.25),
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512), #线性变换乘权重矩阵计算
            nn.ReLU(inplace=True), #relu激活函数
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

class ModelC_Large(nn.Module):
    """
    大型 CNN：4 层卷积，3×3 卷积核，更多滤波器
    参数量约 3.2M
    """
    def __init__(self, num_classes=7):
        super().__init__()
        # 卷积层1: 1→64, 3×3 conv
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), #归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 48->24
            nn.Dropout2d(0.25), #0.25丢弃 鲁棒性提升
        )
        # 卷积层2: 64→128, 3×3 conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), #归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 24->12
            nn.Dropout2d(0.25),
        )
        # 卷积层3: 128→256, 3×3 conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 12->6
            nn.Dropout2d(0.25),
        )
        # 卷积层4: 256→512, 3×3 conv
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 归一化方便激活
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 6->3
            nn.Dropout2d(0.25),
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 512), #线性变换乘权重矩阵计算
            nn.ReLU(inplace=True), #relu激活函数
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

