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
import math
import torch
import torch.nn as nn
import einops


class PosEmbedding2d(nn.Module):
    def __init__(self, dim: int, height: int, width: int):
        super().__init__()
        assert dim % 4 == 0, f"Dim must be divisible by 4, got dim {dim}"

        pe = torch.zeros(dim, height, width)

        half_dim = dim // 2
        div_term = torch.exp(torch.arange(0., half_dim, 2) *
                             -(math.log(10000.0) / half_dim))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:half_dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:half_dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[half_dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[half_dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        self.pos_emb = einops.rearrange(pe, 'c h w -> (h w) c')

        self.mlp = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim)
        )

    def forward(self, x: torch.Tensor):
        pos_emb = self.pos_emb.to(x.device)
        pos_emb = self.mlp(pos_emb)

        return x + pos_emb


if __name__ == "__main__":
    ipt = torch.randn((64, 32*32, 16))
    emb = PosEmbedding2d(16, 32, 32)
    print("ipt:", ipt.shape)            # torch.Size([64, 1024, 16])
    print("sin:", emb(ipt).shape)       # torch.Size([64, 1024, 16])
