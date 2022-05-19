import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock, self).__init__()      
        
        if in_channels == out_channels:
            stride = 1
            self.shortcut = nn.Identity()
        else:
            stride = 2
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)

        padding = (kernel_size // 2, kernel_size // 2)
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv1_bn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv2_bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv1_bn(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.conv2_bn(out)

        x = self.shortcut(x)
        out = torch.add(out, x)
        out = self.activation(out)

        return out

