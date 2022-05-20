import torch
import torch.nn as nn
from layers import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, input_shape: tuple, blocks: list = [64, 64, 128, 128, 256, 256], kernel_size: int = 3):
        """
        Make a ResNet with residual blocks.

        Args:
            input_shape (tuple): Shape of the input image (n_channels, height, width)
            blocks (list): List of channels, where each number represents a residual block
            kernel_size (int): kernel size for the convolutional layers
        """
        super().__init__()

        n_channels, height, width = input_shape

        self.conv1 = nn.Conv2d(n_channels, blocks[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        in_channels = blocks[0]
        for block_channels in blocks:
            self.blocks.append(
                ResidualBlock(in_channels, block_channels)
            )
            in_channels = block_channels

    def forward(self, x):
        print("start:", x.shape)
        x = self.conv1(x)
        print("after first conv:", x.shape)
        for block in self.blocks:
            print(x.shape)
            x = block(x)

        return x


if __name__ == "__main__":

    img_shape = (3, 128, 128)
    model = ResNet(img_shape)

    x = torch.randn((32, *img_shape))

    print("Input shape:", x.shape)
    print(model)
    out = model(x)

    print("Output shape:", out.shape)

