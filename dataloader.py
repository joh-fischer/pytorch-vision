import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10:
    def __init__(self, batch_size: int = 16):
        """
        Wrapper to load, preprocess and deprocess CIFAR-10 dataset.

        Args:
            batch_size (int): Batch size, default: 16.
        """
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        self.batch_size = batch_size

        self.mean = [0.491, 0.482, 0.446]
        self.std = [1, 1, 1]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.mean, self.std)
        ])

        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                      transform=self.transform)
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        self.val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                    transform=transforms.ToTensor())
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    @property
    def train(self):
        """ Return training dataloader. """
        return self.train_loader

    @property
    def val(self):
        """ Return validation dataloader. """
        return self.val_loader
