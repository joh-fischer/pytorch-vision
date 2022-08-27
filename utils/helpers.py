import os
import torch


def timer(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def load_checkpoint(model, optimizer, filepath, device):
    """ Load model checkpoint """
    if os.path.isfile(filepath):
        ckpt = torch.load(filepath, map_location=device)
        start_epoch = ckpt['epoch'] + 1
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print("{:<18}: {} (epoch: {})".format('Loaded checkpoint', filepath, start_epoch))
    else:
        raise ValueError("Checkpoint path '{}' does not exist!".format(filepath))

    return model, optimizer, start_epoch


def save_checkpoint(model, optimizer, ckpt_dir, logger):
    epoch = logger.running_epoch
    state = {'epoch': epoch, 'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'loss': logger.epoch['loss'].avg, 'val_loss': logger.epoch['val_loss'].avg,
             'acc': logger.epoch['acc'].avg, 'val_acc': logger.epoch['val_acc'].avg}
    filename = os.path.join(ckpt_dir, f'e{epoch + 1}.pt')
    print(f"Save checkpoint to '{filename}'")
    torch.save(state, filename)