# MIT License Copyright (c) 2022 joh-fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
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
            stride = (1, 1)
            self.shortcut = nn.Identity()
        else:
            stride = (2, 2)
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), stride=stride)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.conv1_bn = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2_bn = nn.BatchNorm2d(out_channels)

        self.activation = nn.ReLU()

    def forward(self, x):
        identity = self.shortcut(x)

        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)

        x = x + identity
        x = self.activation(x)

        return x
