import torch
import torch.nn as nn
from layers import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, in_channels: int = 3, blocks: list = [16, 16, 32, 32, 64, 64], kernel_size: int = 3):
        """
        Create a Residual Network according to the paper of He et al. (2016) [https://arxiv.org/abs/1512.03385]
        for CIFAR-10.

        Args:
            in_channels (int): Number of input channels, e.g. 3 for RGB images.
            blocks (list): List of channels, where each number represents a residual block and its respective number of
                filters.
            kernel_size (int): Kernel size for the convolutional layers, default: 3.
        """
        super().__init__()

        self.entry_conv = nn.Conv2d(in_channels, blocks[0], kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))

        self.residual_blocks = nn.ModuleList()

        running_in_channels = blocks[0]
        for block_channels in blocks:
            self.residual_blocks.append(
                ResidualBlock(running_in_channels, block_channels, kernel_size=kernel_size)
            )
            running_in_channels = block_channels
        
    def forward(self, x):
        x = self.entry_conv(x)

        for block in self.residual_blocks:
            x = block(x)

        return x


if __name__ == "__main__":

    img_shape = (3, 32, 32)
    model = ResNet()

    img_batch = torch.randn((32, *img_shape))

    print("Input shape:", img_batch.shape)
    print("forward pass...")
    out = model(img_batch)

    print("Output shape:", out.shape)

