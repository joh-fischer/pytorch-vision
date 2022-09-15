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
import einops
import torch
import torch.nn as nn

from models.hip.hip_block import HiPBlock
from models.hip.embedding import PosEmbedding2d


class HierarchicalPerceiver(nn.Module):
    def __init__(self, start_dim: int, blocks_cfgs: list,
                 in_channels: int = 3, image_size: int = 32, n_classes: int = 10):
        """
        Hierarchical Perceiver (https://arxiv.org/abs/2202.10890).

        Args:
            start_dim: Dimension to which the rgb color channel gets projected to.
            blocks_cfgs: List of dictionaries containing the input arguments for
                the respective HiP-Block.
            in_channels: Number of input channels (usually 3).
            image_size: Size of input image (cifar10: 32).
            n_classes: Number of output classes.
        """
        super().__init__()

        self.projection = nn.Linear(in_channels, start_dim)

        # positional embedding
        self.pos_embed = PosEmbedding2d(start_dim, image_size, image_size, mlp=True)

        self.encoder_blocks = nn.ModuleList([
            HiPBlock(**block_cfg)
            for block_cfg in blocks_cfgs
        ])

        last_dim = blocks_cfgs[-1]['latent_dim']
        self.head = nn.Sequential(
            nn.LayerNorm(last_dim),
            nn.Linear(last_dim, n_classes)
        )

    def forward(self, x: torch.Tensor):
        # reshape image to sequence
        x = einops.rearrange(x, 'b c h w -> b (h w) c')

        x = self.projection(x)

        x = self.pos_embed(x)

        for block in self.encoder_blocks:
            x = block(x)

        # global average pooling like in SimpleViT?
        x = x.mean(dim=1)

        out = self.head(x)

        return out


if __name__ == "__main__":
    ipt = torch.randn((64, 3, 32, 32))

    cfg = [
        {'input_dim': 16, 'groups': 2, 'n_latents': 128, 'latent_dim': 64},
        {'input_dim': 64, 'groups': 2, 'n_latents': 128, 'latent_dim': 64}
    ]
    hip = HierarchicalPerceiver(start_dim=16, blocks_cfgs=cfg)

    print("ipt:", ipt.shape)            # torch.Size([64, 3, 32, 32])
    print("out:", hip(ipt).shape)       # torch.Size([64, 10])
