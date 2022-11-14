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

from models.convnext.spatial_layernorm import LayerNorm2D


class SpatialDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, eps: float = 1e-6):
        super().__init__()
        self.layer_norm = LayerNorm2D(in_channels, eps)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        x = self.layer_norm(x)
        x = self.downsample(x)

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 16, 64, 64))
    dsc = SpatialDownsample(16)

    print("input:", ipt.shape)          # torch.Size([8, 16, 64, 64])
    print("output:", dsc(ipt).shape)    # torch.Size([8, 16, 32, 32])
