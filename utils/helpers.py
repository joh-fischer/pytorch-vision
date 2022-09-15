import os
import torch
from models import ResNet, ViT


def get_model(name, config):
    if name == "resnet":
        return ResNet(**config)
    if name == "vit":
        return ViT(**config)
    if name == "hip":
        return
    else:
        raise ValueError(f"Model '{name}' not implemented yet!")


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def load_checkpoint(model, filepath, device):
    """ Load model checkpoint """
    if os.path.isfile(filepath):
        ckpt = torch.load(filepath, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model_state_dict'])
        print("{:<18}: {} (epoch: {})".format('Loaded checkpoint', filepath, start_epoch))
    else:
        raise ValueError("Checkpoint path '{}' does not exist!".format(filepath))

    return model, start_epoch


def save_checkpoint(model, ckpt_dir, logger):
    epoch = logger.running_epoch
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'loss': logger.epoch['loss'].avg, 'val_loss': logger.epoch['val_loss'].avg,
             'acc': logger.epoch['acc'].avg, 'val_acc': logger.epoch['val_acc'].avg}
    filename = os.path.join(ckpt_dir, 'ckpt.pt')
    print(f"Save checkpoint to '{filename}'")
    torch.save(state, filename)


def count_parameters(model, return_int: bool = False):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if return_int:
        return n_params

    return f'{n_params:,}'
