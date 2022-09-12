# Image Classification Models

Implementation of a few CIFAR-10 image classification models in PyTorch. 


## Results

Each model is trained for 20 epochs with learning rate 0.001, Adam optimizer and a batch
size of 128 on a GTX 1050 Ti gpu.


|                   Paper                    |          Code           |  Params   | Accuracy |
|:------------------------------------------:|:-----------------------:|:---------:|:--------:|
| [ResNet](https://arxiv.org/abs/1512.03385) | [resnet](models/resnet) |  175,594  |  95.2%   |
|  [ViT](https://arxiv.org/abs/2010.11929)   |    [vit](models/vit)    | 1,200,906 |  94.4%   |    

## Usage

You can train a model with

```
python3 main.py resnet --name exp1 --epochs 2
```

A list of supported models can be found in the results section.

### ResNet

He et al. ([2016](https://arxiv.org/abs/1512.03385)) introduced skip connections
to build deeper models.

```python
from models import ResNet

model = ResNet()

x = torch.randn((64, 3, 32, 32))
model(x).shape      # [64, 10] 
```

### ViT

Dosovitskiy et al. ([2020](https://arxiv.org/abs/2010.11929)) propose the Vision Transformer (ViT), which
simply applies the NLP Transformer Encoder to images.

```python
from models import ViT

model = ViT()

x = torch.randn((64, 3, 32, 32))
model(x).shape      # [64, 10] 
```
