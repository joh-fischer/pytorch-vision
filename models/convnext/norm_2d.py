import torch
import torch.nn as nn
import einops


class LayerNorm2D(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim, eps)

    def forward(self, x: torch.Tensor):
        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.layer_norm(x)
        x = einops.rearrange(x, 'b h w c -> b c h w')

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 16, 32, 32))
    dsc = LayerNorm2D(16)

    print("input:", ipt.shape)          # torch.Size([8, 16, 32, 32])
    print("output:", dsc(ipt).shape)    # torch.Size([8, 16, 32, 32])
