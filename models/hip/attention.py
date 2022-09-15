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


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, dim: int = 64, heads: int = 4,
                 cross_attn: bool = False):
        """
        Multi-head attention module as described in "Attention Is All
        You Need" (https://arxiv.org/abs/1706.03762).

        Args:
            input_dim: Dimension of input.
            dim: Dimension of keys, queries, values.
            heads: Number of heads.
            cross_attn: If true, applies cross-attention. Otherwise,
                it is normal self-attention.
        """
        super().__init__()

        self.heads = heads
        self.scale = dim ** -0.5

        latent_dim = dim if cross_attn else input_dim

        self.to_q = nn.Linear(latent_dim, dim * heads, bias=False)

        self.to_k = nn.Linear(input_dim, dim * heads, bias=False)
        self.to_v = nn.Linear(input_dim, dim * heads, bias=False)

        self.unify_heads = nn.Linear(dim * heads, latent_dim)

    def forward(self, x: torch.Tensor, latents: torch.Tensor = None):
        k = self.to_k(x)
        k = einops.rearrange(k, '... n (h d) -> ... h n d', h=self.heads)
        v = self.to_v(x)
        v = einops.rearrange(v, '... n (h d) -> ... h n d', h=self.heads)

        if latents is None:
            latents = x
        q = self.to_q(latents)
        q = einops.rearrange(q, '... n (h d) -> ... h n d', h=self.heads)

        dot = torch.einsum('... h n d, ... h k d -> ... h n k', q, k) * self.scale

        att = torch.softmax(dot, dim=-1)

        out = torch.einsum('... h d n, ... h n v -> ... h d v', att, v)
        out = einops.rearrange(out, '... h n d -> ... n (h d)')
        out = self.unify_heads(out)

        return out


class SelfAttention(nn.Module):
    def __init__(self, input_dim: int, dim: int = 64, heads: int = 4):
        super().__init__()

        self.attn = MultiHeadAttention(input_dim, dim, heads)

    def forward(self, x: torch.Tensor):
        return self.attn(x)


class CrossAttention(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, heads: int = 8):
        """
        Applies cross attention between latent and input array as
        described in https://arxiv.org/abs/2103.03206.
        """
        super().__init__()

        self.attn = MultiHeadAttention(input_dim, latent_dim, heads, cross_attn=True)

    def forward(self, x: torch.Tensor, latents: torch.Tensor):
        """
        Args:
            x: Byte tensor of shape [bs, groups, group_len, c]
            latents: Latent tensor of shape [groups, n_latents, latent_dim]
        Returns:
            Tensor of shape [bs, groups, n_latents, latent_dim]
        """
        return self.attn(x, latents)


if __name__ == "__main__":
    bs = 32

    print("Self-Attention")
    ipt = torch.randn((bs, 16, 64))
    sa = SelfAttention(64, 128)
    print("\tin:", ipt.shape)             # torch.Size([bs, 16, 64])
    print("\tout:", sa(ipt).shape)        # torch.Size([bs, 16, 64])

    print("Cross-Attention")
    groups = 4
    K, D = (128, 128)
    lat = torch.randn((groups, K, D))

    M, C = (1024, 3)
    ipt = torch.randn((bs, groups, M // groups, C))
    ca = CrossAttention(C, D, heads=6)
    print("\tlatents:", lat.shape)          # torch.Size([4, 128, 128])
    print("\tin:", ipt.shape)               # torch.Size([32, 4, 256, 3])
    print("\tout:", ca(ipt, lat).shape)     # torch.Size([32, 4, 128, 128])
