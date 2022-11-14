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
