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


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int,
                 n_classes: int,
                 n_output_query_channels: int = 256,
                 n_output_queries: int = 1,
                 cross_attn_heads: int = 4):
        """
        Cross-attending classification head for Hierarchical Perceiver.
        Args:
            in_dim: Input dimension (output dimension or latent_dim of encoder)
            n_classes: Number of output classes.
            n_output_query_channels: Number of output query channels.
            n_output_queries: Number of output queries (default: 1).
            cross_attn_heads: Number of cross-attention heads.
        """
        super().__init__()
        self.output_query = nn.Parameter(torch.randn((n_output_queries, n_output_query_channels)))
        self.cross_attn = CrossAttention(in_dim, n_output_query_channels, heads=cross_attn_heads)
        self.to_logits = nn.Linear(n_output_query_channels, n_classes)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Output of HiP encoder, with shape (bs, n_latents, latent_dim)
        Returns:
            logits: Classification output, with shape (bs, n_classes)
        """
        bs = x.shape[0]
        output_query = einops.repeat(self.output_query, '... -> b ...', b=bs)   # (bs, 1, n_output_query_channels)

        # cross-attend to latents
        x = self.cross_attn(x, output_query)

        logits = self.to_logits(x).squeeze(dim=1)

        return logits


if __name__ == "__main__":
    bs = 32

    ipt = torch.randn((bs, 64, 1024))
    head = ClassificationHead(1024, n_classes=1000)

    print("ipt:", ipt.shape)            # torch.Size([32, 64, 1024])
    print("out:", head(ipt).shape)      # torch.Size([32, 1000])
