import yaml
import torch
from utils.helpers import count_parameters

from models import ResNet, ViT, HierarchicalPerceiver

in_channels = 3


ipt = torch.randn((64, in_channels, 32, 32))
print("input:", ipt.shape)

cfg = yaml.load(open('configs/resnet.yaml', 'r'), Loader=yaml.Loader)
resnet = ResNet(**cfg)
print(f"ResNet\n\tparams: {count_parameters(resnet)}\n\toutput: {resnet(ipt).shape}")

cfg = yaml.load(open('configs/vit.yaml', 'r'), Loader=yaml.Loader)
vit = ViT(**cfg)
print(f"ViT\n\tparams: {count_parameters(vit)}\n\toutput: {vit(ipt).shape}")

cfg = yaml.load(open('configs/hip.yaml', 'r'), Loader=yaml.Loader)
hip = HierarchicalPerceiver(**cfg)
print(f"HiP\n\tparams: {count_parameters(hip)}\n\toutput: {hip(ipt).shape}")
