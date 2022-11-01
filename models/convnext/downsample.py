import torch
import torch.nn as nn

from models.convnext.norm_2d import LayerNorm2D


class SpatialDownsample(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.layer_norm = LayerNorm2D(channels, eps)
        self.downsample = nn.Conv2d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.layer_norm(x)
        x = self.downsample(x)

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 16, 64, 64))
    dsc = SpatialDownsample(16)

    print("input:", ipt.shape)          # torch.Size([8, 16, 64, 64])
    print("output:", dsc(ipt).shape)    # torch.Size([8, 16, 32, 32])
