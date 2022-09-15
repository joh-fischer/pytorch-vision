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
import einops

from models.hip.attention import CrossAttention
from models.hip.transformer import TransformerBlock


class HiPBlock(nn.Module):
    def __init__(self, input_dim: int, groups: int,
                 n_latents: int, latent_dim: int,
                 sa_layers: int = 4, heads: int = 4,
                 dropout: float = 0.2):
        """
        Hierarchical Perceiver Block.

        Args:
            input_dim: Input dimension to the perceiver block.
            groups: Number of groups for this block.
            n_latents: Number of latent vectors per group.
            latent_dim: Dimension of the latent vectors.
            sa_layers: Number of self-attention layers.
            heads: Number of heads for self- and cross-attention.
            dropout: Dropout rate.
        """
        super().__init__()

        self.groups = groups

        self.latents = nn.Parameter(torch.randn((self.groups, n_latents, latent_dim)))

        self.cross_attn = CrossAttention(input_dim, latent_dim, heads)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, heads, dropout)
            for _ in range(sa_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Input tensor with shape [bs, seq_len, dim]
        """
        # group input
        x = einops.rearrange(x, 'b (g n) c -> b g n c', g=self.groups)

        latents = self.cross_attn(x, self.latents)

        for transformer_block in self.transformer_blocks:
            latents = transformer_block(latents)

        # merge input
        out = einops.rearrange(latents, 'b g n c -> b (g n) c')

        return out


if __name__ == "__main__":
    bs = 32

    ipt = torch.randn((bs, 64*64, 16))
    hip_block = HiPBlock(input_dim=16, groups=8,
                         n_latents=32, latent_dim=64)

    print("ipt:", ipt.shape)                # torch.Size([32, 4096, 32])
    print("out:", hip_block(ipt).shape)     # torch.Size([32, 256, 64])
