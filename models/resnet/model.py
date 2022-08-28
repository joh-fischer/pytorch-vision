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
import torch
import torch.nn as nn
from models.resnet.layers import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, in_channels: int = 3, blocks: list = [16, 16, 32, 32, 64, 64], kernel_size: int = 3,
                 n_classes: int = 10, first_conv_cfg: dict = None):
        """
        Create a Residual Network according to the paper of He et al. (2016) [https://arxiv.org/abs/1512.03385].

        Args:
            in_channels (int): Number of input channels, e.g. 3 for RGB images.
            blocks (list): List of channels, where each number represents a residual block and
            its respective number of filters.
            kernel_size (int): Kernel size for the convolutional layers, default: 3.
            n_classes (int): If not None, classification head is included (global average pooling
                and output layer).
            first_conv_cfg (dict): Dictionary of key-value pairs for first convolutional layer.
                If not specified it is {'kernel_size': 3, 'padding': 1, 'stride': 1}.
        """
        super().__init__()
        self.n_classes = n_classes

        conv_config = first_conv_cfg if first_conv_cfg else {'kernel_size': 3, 'padding': 1, 'stride': 1}
        self.first_conv = nn.Conv2d(in_channels, blocks[0], **conv_config)

        self.residual_blocks = nn.ModuleList()
        running_in_channels = blocks[0]
        for block_channels in blocks:
            self.residual_blocks.append(
                ResidualBlock(running_in_channels, block_channels, kernel_size=kernel_size)
            )
            running_in_channels = block_channels

        if n_classes:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.out = nn.Linear(blocks[-1], self.n_classes)

    def forward(self, x):
        x = self.first_conv(x)

        for block in self.residual_blocks:
            x = block(x)

        if self.n_classes:
            x = self.avgpool(x)
            x = x.view(x.shape[0], -1)
            x = self.out(x)

        return x


if __name__ == "__main__":
    img_shape = (3, 32, 32)
    model = ResNet()

    img_batch = torch.randn((64, *img_shape))
    out = model(img_batch)

    print("Input shape:", img_batch.shape)
    print("Output shape:", out.shape)
