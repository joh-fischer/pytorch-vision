import os
import time
import yaml
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from model import ResNet
from dataloader import CIFAR10
from utils.logger import Logger


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
parser.add_argument('--config', default='config.yml',
                    metavar='PATH', help='Path to model config file (default: config.yml)')
parser.add_argument('--gpus', default=0, type=int,
                    nargs='+', metavar='GPUS', help='If GPU(s) available, which GPU(s) to use for training.')
parser.add_argument('--ckpt-save', default=True, action=argparse.BooleanOptionalAction,
                    dest='save_checkpoint', help='Save checkpoints to folder')
parser.add_argument('--load-ckpt', default=None, metavar='PATH',
                    dest='load_checkpoint', help='Load model checkpoint and continue training')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Evaluate model on test set')

writer = None
logger = Logger(LOG_DIR)


def main():
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    running_log_dir = os.path.join(LOG_DIR, args.name, f'{TIMESTAMP}')
    running_ckpt_dir = os.path.join(CHECKPOINT_DIR, args.name, f'{TIMESTAMP}')
    print("{:<16}: {}".format('logdir', running_log_dir))
    print("{:<16}: {}".format('ckpt_dir', running_ckpt_dir))

    # TODO: include multi-gpu training
    gpus = str(args.gpus) if isinstance(args.gpus, int) else ','.join([str(g) for g in args.gpus])
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print("{:<16}: {}\n".format('device', device))

    # read config file
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    # create model
    model = ResNet(**cfg)
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint))
    model.to(device)

    # define loss function and optimizer
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # load data
    data = CIFAR10(args.batch_size)


if __name__ == "__main__":
    main()
