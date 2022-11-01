import torch
import torch.nn as nn

from models.convnext.ds_conv import DepthwiseSeparableConv
from models.convnext.norm_2d import LayerNorm2D


class ConvNeXtBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 kernel_size: int = 7,
                 widening_factor: int = 4,
                 eps: float = 1e-6):
        super().__init__()
        padding = kernel_size // 2
        self.ds_conv = DepthwiseSeparableConv(channels, channels, kernel_size, padding=padding)
        self.norm = LayerNorm2D(channels, eps)
        self.conv1 = nn.Conv2d(channels, channels * widening_factor, kernel_size=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv2d(channels * widening_factor, channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.ds_conv(x)
        x = self.norm(x)

        x = self.conv1(x)   # inverted bottleneck
        x = self.gelu(x)

        x = self.conv2(x)   # reduce channels

        x += residual

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 16, 32, 32))
    dsc = ConvNeXtBlock(16)

    print("input:", ipt.shape)          # torch.Size([8, 3, 64, 64])
    print("output:", dsc(ipt).shape)    # torch.Size([8, 1, 64, 64])
