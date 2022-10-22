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

from models.hip.attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, dim_head: int = 32, heads: int = 4,
                 dropout: float = 0., widening_factor: int = 4):
        """
        Transformer Encoder block as described in https://arxiv.org/abs/2010.11929.
        Args:
            dim: Input dimension of transformer block.
            dim_head: Dimension of keys, queries, values per head.
            heads: Number of heads for attention.
            dropout: p for dropout layers (default: 0).
            widening_factor: MLP hidden dim widening factor (default: 4).
        """
        super().__init__()
        self.attn = SelfAttention(dim, dim_head, heads)

        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * widening_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * widening_factor, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        x = self.attn(x) + x
        x = self.mlp(x) + x

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 16, 64))
    tblock = TransformerBlock(64)
    print("transformer block in:", ipt.shape)             # torch.Size([8, 16, 64])
    print("transformer block out:", tblock(ipt).shape)    # torch.Size([8, 16, 64])
