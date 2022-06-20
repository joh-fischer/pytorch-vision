import os
import time
import yaml
import argparse
from tqdm import tqdm
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
parser.add_argument('--log-save-interval', default=5, type=int,
                    metavar='N', help="Interval in which logs are saved to disk (default: 5)")

writer = None
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
    logger = Logger(running_log_dir)
    logger.log_hparams({**cfg, **vars(args)})
    global writer
    writer = SummaryWriter(running_log_dir)

    # setup devices
    # TODO: include multi-gpu training
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print("{:<16}: {}\n".format('device', device))

    # load data
    data = CIFAR10(args.batch_size)

    # create model
    model = ResNet(**cfg)
    if args.load_checkpoint:
        model.load_state_dict(torch.load(args.load_checkpoint))
    model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.evaluate:
        # TODO: evaluate
        pass

    # start run
    t_start = time.time()
    for epoch in range(args.epochs):
        logger.init_epoch()

        print(f"Epoch [{epoch+1} / {args.epochs}]")

        # TODO: add accuracy (top-1 and top-5)
        train(model, data.train, criterion, optimizer, device)
        writer.add_scalar('Loss/train', logger.epoch['loss'].avg, epoch)

        validate(model, data.val, criterion, device)
        writer.add_scalar('Loss/val', logger.epoch['val_loss'].avg, epoch)

        print(f"loss: {logger.epoch['loss'].avg:.3f} - val_loss: {logger.epoch['val_loss'].avg:.3f}")

        if args.save_checkpoint:
            torch.save(model.state_dict(), os.path.join(running_ckpt_dir, f'e{epoch}.ckpt'))

        if epoch % args.log_save_interval == 0:
            logger.save()

    logger.save()

    t_end = time.time() - t_start
    print(f"Total training time: {t_end:.4f} s")

    # TODO: write to summary file
    write_summary(args.name, running_log_dir, running_ckpt_dir,
                  args.config, args.epochs, t_end,
                  logger.epoch['loss'].avg, logger.epoch['val_loss'].avg)


def train(model, train_loader, criterion, optimizer, device):
    model.train()

    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.to(device), y.to(device)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.log_metrics({'loss': loss.item()}, phase='train', aggregate=True, n=x.shape[0])


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()

    for x, y in tqdm(val_loader, desc="Validation"):
        x, y = x.to(device), y.to(device)

        output = model(x)
        loss = criterion(output, y)

        logger.log_metrics({'val_loss': loss.item()}, phase='val', aggregate=True, n=lead.shape[0])


def write_summary(*infos):
    with open(SUMMARY_FILE, 'a') as sf:
        sf.write(','.join([str(i) for i in infos]) + '\n')


if __name__ == "__main__":
    main()
