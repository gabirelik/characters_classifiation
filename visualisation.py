import numpy as np
import matplotlib.pyplot as plt


def get_discrete_bins(x):
    return np.arange(0, x.max() + 1.5) - 0.5


def plot_histogram(x, title, x_label, y_label, bins=None, center_ticks=False):
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.hist(x, bins)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if center_ticks:
        ax.set_xticks(bins + 0.5)

    plt.show()


def show_samples(x):
    fig, axs = plt.subplots(4, 4, figsize=(4, 4))
    [ax.set_axis_off() for ax in axs.flatten()]
    [ax.imshow(img) for ax, img in zip(axs.flatten(), x)]
    plt.show()
