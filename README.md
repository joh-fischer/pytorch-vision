# ResNet Implementation

Implementing CIFAR10 classifying ResNet trained on GPU.

## Todo
- GPU access
  - multiple GPU support
  - to(device)
- argparse
  - set hyperparameters
- progress bar as wright
- TensorBoard support
- Custom Logger
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
- data preprocessing
  - per pixel mean subtracted
  - 4 pixel padding and center crop
  - horizontal flip

## Description
- architecture as in ResNet paper for CIFAR-10
  - filters (16, 32, ...)
  - avgpool2d (GAP)

## Links
- https://github.com/pytorch/examples/blob/main/imagenet/main.py
- https://github.com/matthias-wright/cifar10-resnet