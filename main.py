import os
import time
import yaml
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn

from resnet import ResNet
from utils.logger import Logger
from dataloader import CIFAR10
from utils import load_checkpoint, timer, save_checkpoint

CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'
TIMESTAMP = datetime.now().strftime('%y-%m-%d_%H%M%S')
SUMMARY_FILE = os.path.join(LOG_DIR, 'models_summary.csv')

parser = argparse.ArgumentParser(description="PyTorch ResNet Training")
parser.add_argument('--name', '-n', default='',
                    type=str, metavar='NAME', help='Model name and folder where logs are stored')
parser.add_argument('--epochs', default=2,
                    type=int, metavar='N', help='Number of epochs to run (default: 2)')
parser.add_argument('--batch-size', default=8, metavar='N',
                    type=int, help='Mini-batch size (default: 8)')
parser.add_argument('--lr', default=0.01,
                    type=float, metavar='LR', help='Initial learning rate (default: 0.01)')
parser.add_argument('--config', default='config.yaml',
                    metavar='PATH', help='Path to model config file (default: config.yaml)')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--ckpt-save', default=True, action=argparse.BooleanOptionalAction,
                    dest='save_checkpoint', help='Save checkpoints to folder')
parser.add_argument('--load-ckpt', default=None, metavar='PATH',
                    dest='load_checkpoint', help='Load model checkpoint and continue training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Evaluate model on test set')
parser.add_argument('--log-save-interval', default=5, type=int, metavar='N',
                    dest='save_interval', help="Interval in which logs are saved to disk (default: 5)")

logger = Logger(LOG_DIR)


def main():
    # parse args
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # read config file
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # setup paths and logging
    running_log_dir = os.path.join(LOG_DIR, args.name, f'{TIMESTAMP}')
    running_ckpt_dir = os.path.join(CHECKPOINT_DIR, args.name, f'{TIMESTAMP}')
    print("{:<16}: {}".format('logdir', running_log_dir))
    print("{:<16}: {}".format('ckpt_dir', running_ckpt_dir))

    if args.save_checkpoint and not os.path.exists(running_ckpt_dir):
        os.makedirs(running_ckpt_dir)

    global logger
    logger = Logger(running_log_dir, tensorboard=True)

    # setup GPU
    # TODO: include multi-gpu training
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print("{:<16}: {}\n".format('device', device))

    # load data
    data = CIFAR10(args.batch_size)

    # create model, loss function and optimizer
    model = ResNet(**cfg)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss().to(device)

    if args.load_checkpoint:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, args.load_checkpoint, device)
        args.epochs += start_epoch
    else:
        start_epoch = 0

    if args.evaluate:
        logger.init_epoch()
        validate(model, data.val, criterion, device)
        print("\n{:>8}: {:.4f} - {:>8}: {:.4f}".format('val_loss', logger.epoch['val_loss'].avg,
                                                       'val_acc', 100. * logger.epoch['val_acc'].avg))
        return

    # start run
    logger.log_hparams({**cfg, **vars(args)})
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        logger.init_epoch(epoch)
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(model, data.train, criterion, optimizer, device)
        validate(model, data.val, criterion, device)

        # log to tensorboard
        logger.tensorboard.add_scalars('Accuracy', {'train_acc': logger.epoch['acc'].avg,
                                                    'val_acc': logger.epoch['val_acc'].avg}, epoch)
        logger.tensorboard.add_scalars('Loss', {'train_loss': logger.epoch['loss'].avg,
                                                'val_loss': logger.epoch['val_loss'].avg}, epoch)

        # output progress
        print("\n{:>8}: {:.4f} - {:>8}: {:.4f} - {:>4}: {:.2f} - {:>4}: {:.2f}".format(
            'loss', logger.epoch['loss'].avg, 'val_loss', logger.epoch['val_loss'].avg,
            'acc', 100. * logger.epoch['acc'].avg, 'val_acc', 100. * logger.epoch['val_acc'].avg))

        # save logs and checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                save_checkpoint(model, optimizer, running_ckpt_dir, logger)

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)

        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        acc = correct / y.shape[0]

        logger.log_metrics({'loss': loss, 'acc': acc},
                           phase='train', aggregate=True, n=x.shape[0])


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    for x, y in tqdm(val_loader, desc="Validation"):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)

        loss = criterion(y_hat, y)

        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        acc = correct / y.shape[0]
        logger.log_metrics({'val_loss': loss, 'val_acc': acc},
                           phase='val', aggregate=True, n=x.shape[0])


if __name__ == "__main__":
    main()
