# Image Classification Models

Implementation of a few popular vision models in PyTorch. 

## Usage

You can train the models with

```
python3 main.py resnet --name exp1 --epochs 60 --batch-size 256 --warmup-epochs 10
```

A list of supported models can be found in the results section (*code* column).


## Results

Results of some trainings on `CIFAR10`. I train the models with AdamW optimizer for 90 pochs using
a cosine decay learning rate scheduler and 10 epochs linear warm-up. Please note, that the reported
accuracies are far from what is possible with those models, as I just train them for a couple of
epochs and don't finetune them at all. ;)


|                           Paper                            |            Code             |  Params   | Accuracy |
|:----------------------------------------------------------:|:---------------------------:|:---------:|:--------:|
|         [ResNet](https://arxiv.org/abs/1512.03385)         |   [resnet](models/resnet)   |  175,594  |  89.1%   |
|        [ConvNeXt](https://arxiv.org/abs/2201.03545)        | [convnext](models/convnext) |  398,730  |  76.2%  |
|          [ViT](https://arxiv.org/abs/2010.11929)           |      [vit](models/vit)      |  305,802  |  68.4%   |
| [Hierarchical Perceiver](https://arxiv.org/abs/2202.10890) |      [hip](models/hip)      | 1,204,138 |  57.6%   |


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
first patchifies the image and then simply applies the NLP Transformer encoder.

```python
from models import ViT

model = ViT()

x = torch.randn((64, 3, 32, 32))
model(x).shape      # [64, 10] 
```

### Hierarchical Perceiver

Carreira et al. ([2022](https://arxiv.org/abs/2202.10890)) improve the efficiency of the Perceiver
by making it hierarchical. For that, the authors propose the HiP-Block which divides the input
sequence into groups and independently applies cross- and self-attention to those groups. Stacking
multiple of those blocks results in the respective hierarchy.

```python
import yaml
from models import HierarchicalPerceiver

cfg = yaml.load(open('configs/hip.yaml', 'r'), Loader=yaml.Loader)
model = HierarchicalPerceiver(**cfg)

x = torch.randn((64, 3, 32, 32))
model(x).shape      # [64, 10] 
```

For this implementation I used standard 2D sinusoidal instead of learned positional embeddings. Furthermore, I only
train on classification with the HiP encoder. However, you can add the decoder simply by editing the
config file of the HiP (add some more blocks with decreasing `latent_dim` and increasing sequence length). Then
the proposed masked auto-encoder pre-training ([MAE](https://arxiv.org/abs/2111.06377)) is quite
straight-forward.


### ConvNeXt

Liu et al. ([2022](https://arxiv.org/abs/2201.03545)) gradually modernize a standard ResNet, by adapting the training
procedure (optimizer, augmentations & regularizations), the macro design (stage compute ratio, patchify stem, depthwise
separable convolutions & inverted bottleneck), and the micro design (GELU, fewer activation and normalization functions,
layer normalization & convolutional downsampling).

```python
from models import ConvNeXt

model = ConvNeXt()

x = torch.randn((64, 3, 224, 224))
model(x).shape      # [64, 1000] 
```

