# Image Classification Models

Implementation of a few CIFAR10 image classification models in PyTorch.

**TODO**:
* describe data preprocessing
* learning rate decay
* describe custom logger
* tensorboard support
* how to start from checkpoint

## ResNet

He et al. ([2016](https://arxiv.org/abs/1512.03385)) introduce skip connections ...

Implementation of residual networks with the same architecture

```python
import torch
from models import ResNet

x = torch.randn((16, 3, 32, 32))

model = ResNet()

model(x).shape      # 
```

## ViT ([Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929))


## Usage



## Links
- https://github.com/pytorch/examples/blob/main/imagenet/main.py
- https://github.com/matthias-wright/cifar10-resnet
