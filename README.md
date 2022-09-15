# Image Classification Models

Implementation of a few CIFAR-10 image classification models in PyTorch. 


## Results

Each model is trained for 20 epochs with learning rate 0.0001, Adam optimizer and a batch
size of 128. Please note, that the results are just from a few runs and models were not
fine-tuned at all. This repo is just for deepening my understanding of those models. ;)


|                           Paper                            |          Code           |  Params   | Accuracy |
|:----------------------------------------------------------:|:-----------------------:|:---------:|:--------:|
|         [ResNet](https://arxiv.org/abs/1512.03385)         | [resnet](models/resnet) |  175,594  |  95.2%   |
|          [ViT](https://arxiv.org/abs/2010.11929)           |    [vit](models/vit)    | 1,200,906 |  94.4%   |
| [Hierarchical Perceiver](https://arxiv.org/abs/2202.10890) |    [hip](models/hip)    | 4,520,970 |  94.4%   |

## Usage

You can train a model with

```
python3 main.py resnet --name exp1 --epochs 2
```

A list of supported models can be found in the results section (column *code*).

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

### Hierarchical Perceiver

Carreira et al. ([2022](https://arxiv.org/abs/2202.10890)) improve the efficiency of Perceiver models
by making it hierarchical. For that, the authors divide the input sequence into groups and independently
apply cross- and self-attention to those groups.

```python
import yaml
from models import HierarchicalPerceiver

cfg = yaml.load(open('configs/hip.yaml', 'r'), Loader=yaml.Loader)
model = HierarchicalPerceiver(**cfg)

x = torch.randn((64, 3, 32, 32))
model(x).shape      # [64, 10] 
```