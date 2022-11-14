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

from models.convnext.block import ConvNeXtBlock
from models.convnext.downsample import SpatialDownsample
from models.convnext.spatial_layernorm import LayerNorm2D


class ConvNeXt(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 96,
                 kernel_size: int = 7, widening_factor: int = 4, drop_path: float = 0.,
                 n_blocks: int = 3, stage_compute_ratios: list = [3, 3, 9, 3],
                 n_classes: int = 1000):
        """
        ConvNeXt model (https://arxiv.org/abs/2201.03545).

        Args:
            in_channels: Number of input channels.
            width: Network width (Default as in the paper: 96).
            kernel_size: Kernel size for the ConvNeXt blocks (Default
                as in the paper: 7).
            widening_factor: Widening factor for the ConvNeXt inverted
                bottleneck (Default as in the paper: 4).
            drop_path: Path dropping probability (Stochastic depth).
            n_blocks: Base number of blocks per stage, which gets multiplied by
                the stage compute ratio per stage.
            stage_compute_ratios: Stage compute ratios per block (Default
                as in the paper: [3, 3, 9, 3]).
            n_classes: Number of classes for the head.
        """
        super().__init__()

        # stem cell
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=4, stride=4),
            LayerNorm2D(width)
        )

        # build stages
        self.stages = nn.ModuleList()

        prev_channels = width
        for it, stage_comp_ratio in enumerate(stage_compute_ratios):
            new_channels = width * (it + 1)

            # downsampling layer
            ds_layer = SpatialDownsample(prev_channels, new_channels) if it != 0 else nn.Identity()

            # blocks
            stage_blocks = [
                ConvNeXtBlock(new_channels, kernel_size, widening_factor, drop_path)
                for _ in range(n_blocks * stage_comp_ratio)
            ]

            self.stages.append(nn.Sequential(ds_layer, *stage_blocks))
            prev_channels = new_channels

        # global average pooling layer + normalization
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # head
        self.head = nn.Sequential(
            nn.LayerNorm(prev_channels),
            nn.Linear(prev_channels, n_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        out = self.head(x)

        return out


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 224, 224))
    model = ConvNeXt()

    print("input:", ipt.shape)              # torch.Size([8, 3, 224, 224])
    print("output:", model(ipt).shape)      # torch.Size([8, 1000])
