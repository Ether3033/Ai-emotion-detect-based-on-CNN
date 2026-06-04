import torch
import torch.nn as nn
#个人练习 函数风格的模型搭建
def _make_block(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(p=0.25),
    )
def _make_connection_block(input_dim, num_classes):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, num_classes),
    )
def ABC_Forward(Model_name, num_classes=7):
    if Model_name == 'A':
        conv1 = _make_block(1, 64)
        conv2 = _make_block(64, 128)
        # 输入 48x48 -> conv1: 24x24 -> conv2: 12x12，通道 128
        fc = _make_connection_block(128 * 12 * 12, num_classes)
        return nn.Sequential(conv1, conv2, fc)
    elif Model_name == 'B':
        conv1 = _make_block(1, 64)
        conv2 = _make_block(64, 128)
        conv3 = _make_block(128, 256)
        # 输入 48x48 -> 24 -> 12 -> 6，通道 256
        fc = _make_connection_block(256 * 6 * 6, num_classes)
        return nn.Sequential(conv1, conv2, conv3, fc)
    elif Model_name == 'C':
        conv1 = _make_block(1, 64)
        conv2 = _make_block(64, 128)
        conv3 = _make_block(128, 256)
        conv4 = _make_block(256, 512)
        # 输入 48x48 -> 24 -> 12 -> 6 -> 3，通道 512
        fc = _make_connection_block(512 * 3 * 3, num_classes)
        return nn.Sequential(conv1, conv2, conv3, conv4, fc)
