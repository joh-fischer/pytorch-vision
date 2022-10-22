import os
import sys
import time
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy

from dataloader import CIFAR10

from utils.logger import Logger
from utils.helpers import load_checkpoint, timer, save_checkpoint
from utils.helpers import get_model
from utils.scheduler import cosine_scheduler

# python3 main.py hip --name run0 --epochs 60 --batch-size 256 --gpus 7 --warmup-epochs 10

parser = argparse.ArgumentParser(description="PyTorch Image Classification")
parser.add_argument('model', choices=['resnet', 'vit', 'hip'], type=str, metavar='NAME',
                    help='Choose model')
parser.add_argument('--name', '-n', default=None, type=str, metavar='NAME',
                    help='Experiment name and folder where logs are stored')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='Number of epochs to run (default: 2)')
parser.add_argument('--batch-size', default=64, metavar='N', type=int,
                    help='Mini-batch size (default: 64)')
parser.add_argument('--gpus', default=0, type=int, nargs='+', metavar='GPUS',
                    help='If GPU(s) available, which GPU(s) to use for training.')

# learning rate stuff
parser.add_argument('--lr', default=0.0001, type=float, metavar='LR',
                    help='Initial learning rate (default: 0.0001)')
parser.add_argument('--use-scheduler', default=True, action=argparse.BooleanOptionalAction,
                    help='Whether to use a cosine scheduler with linear warmup or not.')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='Lower lr bound for cyclic schedulers that hit 0 (1e-6)')
parser.add_argument('--warmup-epochs', type=int, default=1, metavar='N',
                    help='Epochs to warmup LR')

parser.add_argument('--weight-decay', default=0.05, type=float, metavar='WD',
                    help='Weight decay for optimizer (default: 0.05)')
parser.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')

parser.add_argument('--ckpt-save', default=False, action=argparse.BooleanOptionalAction, dest='save_checkpoint',
                    help='Save checkpoints to folder (default: False)')
parser.add_argument('--load-ckpt', default=None, metavar='PATH', dest='load_checkpoint',
                    help='Load model checkpoint and train/evaluate.')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='Evaluate model on test set')
parser.add_argument('--log-save-interval', default=5, type=int, metavar='N', dest='save_interval',
                    help="Interval in which logs are saved to disk (default: 5)")


LOG_DIR = 'runs'
logger = Logger(LOG_DIR)


def main():
    global logger

    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<16}: {}".format(name, val))

    # GPU
    args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
    if len(args.gpus) == 1:
        device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() else 'cpu')
    else:
        raise ValueError('Currently multi-gpu training is not possible')
    print("{:<16}: {}".format('device', device))

    # data
    data = CIFAR10(args.batch_size)

    # model
    cfg_path = os.path.join('configs', args.model + '.yaml')
    if not os.path.exists(cfg_path):
        raise ValueError(f'Model config file "{cfg_path}" does not exist!')
    cfg = yaml.load(open(cfg_path, 'r'), Loader=yaml.Loader)
    model = get_model(args.model, cfg)
    model.to(device)

    # loss function and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    lr_schedule = None
    if args.use_scheduler:
        n_training_steps_per_epoch = data.n_train_samples // args.batch_size
        lr_schedule = cosine_scheduler(args.lr, args.min_lr, args.epochs,
                                       n_training_steps_per_epoch, args.warmup_epochs)

    if args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(device)
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    if args.load_checkpoint:
        model, start_epoch = load_checkpoint(model, args.load_checkpoint, device)
        args.epochs += start_epoch
    else:
        start_epoch = 0

    if args.evaluate:
        logger = Logger(tensorboard=False, create_folder=False)
        logger.init_epoch()
        validate(model, data.val, criterion, device)
        print("\n{:>8}: {:.4f} - {:>8}: {:.4f}".format('val_loss', logger.epoch['val_loss'].avg,
                                                       'val_acc', 100. * logger.epoch['val_acc'].avg))
        return

    # setup paths and logging
    exp_dir = os.path.join(args.model, args.name) if args.name is not None else args.model
    running_log_dir = os.path.join(LOG_DIR, exp_dir)
    print("{:<16}: {}".format('logdir', running_log_dir))
    logger = Logger(running_log_dir, tensorboard=True)
    logger.log_hparams({**cfg, **vars(args)})

    # start run
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        logger.init_epoch(epoch)
        print(f"Epoch [{epoch + 1} / {args.epochs}]")

        train(model, data.train, criterion, optimizer, device, lr_schedule)
        validate(model, data.val, criterion, device)

        # log to tensorboard
        logger.tensorboard.add_scalar('Accuracy/train', logger.epoch['acc'].avg, epoch)
        logger.tensorboard.add_scalar('Accuracy/val', logger.epoch['val_acc'].avg, epoch)
        logger.tensorboard.add_scalar('Loss/train', logger.epoch['loss'].avg, epoch)
        logger.tensorboard.add_scalar('Loss/val', logger.epoch['val_loss'].avg, epoch)

        # output progress
        print(f"{'global step':>12}: {logger.global_train_step} | {'lr':>3}: {lr_schedule[logger.global_train_step]} | "
              f"{'loss':>8}: {logger.epoch['loss'].avg:.4f} - {'val_loss':>8}: {logger.epoch['val_loss'].avg:.4f} - "
              f"{'acc':>4}: {logger.epoch['acc'].avg:.4f} - {'val_acc':>4}: {logger.epoch['val_acc'].avg:.4f}")

        # save logs and checkpoint
        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            logger.save()
            if args.save_checkpoint:
                save_checkpoint(model, running_log_dir, logger)

    elapsed_time = timer(t_start, time.time())
    print(f"Total training time: {elapsed_time}")


def train(model, train_loader, criterion, optimizer, device, lr_schedule):
    model.train()
    loader = tqdm(train_loader, desc="Training")
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        y_hat = model(x)

        if lr_schedule is not None:
            optimizer.param_groups[0]['lr'] = lr_schedule[logger.global_train_step]

        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        acc = correct / y.shape[0]

        logger.log_metrics({'loss': loss, 'acc': acc},
                           phase='train', aggregate=True, n=x.shape[0])

        if logger.global_train_step % 10 == 0:
            loader.set_description(f"step: {logger.global_train_step} | "
                                   f"lr: {optimizer.param_groups[0]['lr']:.1e}")

        logger.global_train_step += 1


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
    try:
        main()
    except KeyboardInterrupt:
        print("Exit training with keyboard interrupt!")
        logger.save()
        sys.exit(0)
