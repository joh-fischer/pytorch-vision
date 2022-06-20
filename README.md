# ResNet Implementation

Implementing CIFAR10 classifying ResNet trained on GPU.

## Usage

Please ensure that you're using Python version 3.9 or above.

## Todo
- include top $k$ accuracy
- start from checkpoint with dict -> state_dict, epoch, ...
- tensorboard
  - visualize examples and assigned class probabilities
  - writer.add_fig(...)
- GPU access
  - multiple GPU support
  - to(device)
- progress bar as wright
- maybe make improved ResNet
  - use swish activation function
  - learning rate decay (first high, then low)
  - 7x7 decomposed into three 3x3 convs (ref literature)
- torchviz visualization
- training
  - learning rate of 0.1 
  - divide by 10 at 32k and 48k iterations
  - end training after 64k iterations
  - batch size 128


## Description
- architecture as in ResNet paper for CIFAR-10
  - filters (16, 32, ...)
  - avgpool2d (GAP)
- Custom Logger
- TensorBoard support
- data preprocessing
  - per pixel mean subtracted
  - 4 pixel padding and center crop
  - horizontal flip



## Links
- https://github.com/pytorch/examples/blob/main/imagenet/main.py
- https://github.com/matthias-wright/cifar10-resnet