import torch
import torch.nn as nn

from models.convnext.block import ConvNeXtBlock


class ConvNeXt(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 96,
                 kernel_size: int = 7,
                 n_blocks: int = 3, stage_compute_ratios: list = [1, 1, 3, 1]):
        super().__init__()

        # stem cell
        self.stem = nn.Conv2d(in_channels, width, kernel_size=4, stride=4)

        # stage 1
        self.stage1 = nn.ModuleList([
            ConvNeXtBlock(width, kernel_size=kernel_size)
            for _ in range(3)
        ])



    def forward(self, x: torch.Tensor):

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 32, 32))
    dsc = ConvNeXt()

    print("input:", ipt.shape)          # torch.Size([8, 16, 32, 32])
    print("output:", dsc(ipt).shape)    # torch.Size([8, 16, 32, 32])