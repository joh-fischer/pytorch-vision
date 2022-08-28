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

#https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
#https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, dim: int = 64, n_heads: int = 4):
        """
        Multi-Head Attention module as described in Attention Is All
        You Need" (https://arxiv.org/abs/1706.03762).

        Args:
            input_dim: Input dimension.
            dim: Dimension of keys, queries, values.
            n_heads: Number of heads for multi-head attention.
        """
        super().__init__()

        self.dim = dim

        self.to_qkv = nn.Linear(input_dim, dim * n_heads * 3, bias=False)
        self.scale = dim ** -0.5

        self.unify_heads = nn.Linear(dim * n_heads, input_dim)

    def forward(self, x: torch.Tensor):
        qkv = self.to_qkv(x)
        q, k, v = einops.rearrange(qkv, 'b t (qkv h d) -> qkv b h t d', qkv=3, d=self.dim)     # [bs, h, t, d]

        dot = torch.einsum('bhtd, bhkd -> bhtk', q, k) * self.scale     # [bs, h, t, t]

        att = torch.softmax(dot, dim=-1)

        out = torch.einsum('bhdt, bhtv -> bhdv', att, v)
        out = einops.rearrange(out, 'b h t d -> b t (h d)')
        out = self.unify_heads(out)

        return out


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int = 3, dim: int = 64):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must by divisible by patch size!"
        self.patch_size = patch_size

        self.projection = nn.Linear(in_channels * patch_size ** 2, dim)

    def forward(self, x: torch.Tensor):
        # patchify images: H x W x C -> N x (P² * C)
        x = einops.rearrange(x, 'b c (h k) (w i) -> b (h w) (k i c)', k=self.patch_size, i=self.patch_size)
        x = self.projection(x)

        return x


class TimeEmbedding(nn.Module):
    def __init__(self, time_emb_dim: int, pos_emb_dim: int, max_len: int = 5000):
        """
        Time embedding for time step t. First, t is embedded using a fixed sinusoidal positional
        embedding, as described in "Attention Is All You Need" (https://arxiv.org/abs/1706.03762),
        followed by a two layer MLP.

        Args:
            time_emb_dim: Dimension of final time embedding
            pos_emb_dim: Embedding dimension for the fixed sinusoidal positional embedding
            max_len: Maximum number of time steps (default: 5000)
        """
        super().__init__()

        self.pos_emb_dim = pos_emb_dim
        self.time_emb_dim = time_emb_dim
        self.max_len = max_len

        # fixed sinusoidal positional embedding
        assert self.pos_emb_dim % 2 == 0, "Embedding dim must be a multiple of 2!"
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        _2i = torch.arange(0, self.pos_emb_dim, 2).float()
        pos_embedding = torch.zeros(self.max_len, self.pos_emb_dim)
        pos_embedding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / self.pos_emb_dim)))
        pos_embedding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / self.pos_emb_dim)))
        self.register_buffer('pos_embedding', pos_embedding, persistent=True)

        # MLP for time embedding
        self.mlp = nn.Sequential(
            nn.Linear(self.pos_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

    def forward(self, t: torch.Tensor):
        t_pos_emb = torch.index_select(self.pos_embedding, 0, t)
        t_emb = self.mlp(t_pos_emb)

        return t_emb


if __name__ == "__main__":
    ipt = torch.randn((8, 6, 64))
    mha = MultiHeadAttention(64)
    print("attention in:", ipt.shape)
    print("attention out:", mha(ipt).shape)

    ipt = torch.randn((8, 3, 32, 32))
    patchify = PatchEmbedding(image_size=32, patch_size=8)
    print("image in:", ipt.shape)
    print("patch emb:", patchify(ipt).shape)

    ipt = torch.randint(0, 10, (8,))
    embed_time = TimeEmbedding(64, 32)
    print("time in:", ipt.shape)
    print("time emb:", embed_time(ipt).shape)
