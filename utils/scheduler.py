import math
import numpy as np
import matplotlib.pyplot as plt


# adapted from https://github.com/facebookresearch/ConvNeXt
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep,
                     warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


if __name__ == "__main__":
    batch_size = 32
    len_dataset_train = 50000

    lr = 4e-3
    min_lr = 1e-6
    epochs = 40
    num_training_steps_per_epoch = len_dataset_train // batch_size
    warmup_epochs = 10

    lr_schedule = cosine_scheduler(lr, min_lr, epochs, num_training_steps_per_epoch, warmup_epochs)
    print(lr_schedule)
    print(len(lr_schedule))

    plt.plot(lr_schedule)
    plt.xlabel("Iteration")
    plt.ylabel("LR")
    plt.show()
