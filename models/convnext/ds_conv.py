import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, output_channels: int, kernel_size: int,
                 padding: int = 0, hidden_multiplier: int = 1):
        super().__init__()
        self.depth_wise = nn.Conv2d(in_channels, in_channels * hidden_multiplier,
                                    kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.point_wise = nn.Conv2d(in_channels * hidden_multiplier, output_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x = self.depth_wise(x)
        x = self.point_wise(x)

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 64, 64))
    dsc = DepthwiseSeparableConv(3, 16, 3, padding=1)

    print("input:", ipt.shape)          # torch.Size([8, 3, 64, 64])
    print("output:", dsc(ipt).shape)    # torch.Size([8, 1, 64, 64])
