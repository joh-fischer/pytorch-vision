import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Residual block as in https://arxiv.org/abs/1512.03385

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size: filter size for convolutions
        """
        super(ResidualBlock, self).__init__()

        padding = (kernel_size // 2, kernel_size // 2)

        if in_channels == out_channels:
            stride = 1
            self.shortcut = nn.Identity()
        else:
            stride = 2
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                      padding=padding, stride=stride)
        
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
        print("adding ", x.shape, "and", out.shape)
        out = torch.add(out, x)
        out = self.activation(out)

        return out
