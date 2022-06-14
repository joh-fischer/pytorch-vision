import torch
import torch.nn as nn
from layers import ResidualBlock


class ResNet(nn.Module):
    def __init__(self, in_channels: int = 3, blocks: list = [16, 16, 32, 32, 64, 64], kernel_size: int = 3,
                 conv_out_size: int = 8, n_classes: int = 10, first_conv_config: dict = None):
        """
        Create a Residual Network according to the paper of He et al. (2016) [https://arxiv.org/abs/1512.03385].

        Args:
            in_channels (int): Number of input channels, e.g. 3 for RGB images.
            blocks (list): List of channels, where each number represents a residual block and its respective number of
                filters.
            kernel_size (int): Kernel size for the convolutional layers, default: 3.
            conv_out_size (int): Output size of final convolutional layer (e.g. 8 for a 8x8), required for global
                average pooling layer.
            n_classes (int): Number of output classes.
            first_conv_config (dict): Dictionary of key-value pairs for first convolutional layer. If not specified
                it is {'kernel_size': 3, 'padding': 1, 'stride': 1}.
        """
        super().__init__()

        conv_config = first_conv_config if first_conv_config else {'kernel_size': 3, 'padding': 1, 'stride': 1}
        self.first_conv = nn.Conv2d(in_channels, blocks[0], **conv_config)

        self.residual_blocks = nn.ModuleList()
        running_in_channels = blocks[0]
        for block_channels in blocks:
            self.residual_blocks.append(
                ResidualBlock(running_in_channels, block_channels, kernel_size=kernel_size)
            )
            running_in_channels = block_channels

        self.avg_pool = nn.AvgPool2d(kernel_size=conv_out_size)

        self.out = nn.Linear(blocks[-1], n_classes)

    def forward(self, x):
        x = self.first_conv(x)

        for block in self.residual_blocks:
            x = block(x)

        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)

        x = self.out(x)

        return x


if __name__ == "__main__":

    img_shape = (3, 32, 32)
    model = ResNet()

    img_batch = torch.randn((32, *img_shape))

    print("Input shape:", img_batch.shape)
    print("forward pass...")
    out = model(img_batch)

    print("Output shape:", out.shape)

