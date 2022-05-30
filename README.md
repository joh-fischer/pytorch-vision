# ResNet Implementation

Implementing CIFAR10 classifying ResNet trained on GPU.

## Todo
- GPU access
  - to(device)
- argparse
  - set hyperparameters
- architecture as in ResNet paper
  - filters (16, 32, ...)
  - avgpool2d (GAP)
  - 2x maxpool
- += residual
- progress bar as wright
- TensorBoard support
- Custom Logger
- maybe make improved ResNet
  - use swish activation function
  - learning rate decay (first high, then low)
  - 7x7 decomposed into three 3x3 convs (ref literature)
- torchviz visualization
