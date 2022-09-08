import torch

from utils.helpers import count_parameters

from models import ResNet

in_channels = 3


ipt = torch.randn((64, in_channels, 32, 32))
print("input:", ipt.shape)

# ResNet
resnet = ResNet()
print(f"ResNet\n\tparams: {count_parameters(resnet)}\n\toutput: {resnet(ipt).shape}")
