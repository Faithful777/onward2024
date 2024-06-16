import warnings

import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as jaccard_score
from matplotlib.ticker import MultipleLocator

def plot_report(history: dict, x_locator_tick: int = 10):
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax2 = ax1.twinx()
    ax1.plot(
        [*history.keys()],
        [x.get("loss") for x in history.values()],
        color="#6ab3a2",
        lw=2,
    )
    ax2.plot(
        [*history.keys()],
        [x.get("base_lr") for x in history.values()],
        color="#3399e6",
        lw=1,
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#6ab3a2", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="#6ab3a2")

    ax2.set_ylabel("Base learning rate", color="#3399e6", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="#3399e6")
    fig.suptitle("Training report", fontsize=18)
    fig.gca().xaxis.set_major_locator(MultipleLocator(x_locator_tick))

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class Upsample(nn.Module):

    def __init__(self,
                 size=None,
                 scale_factor=None,
                 mode='nearest',
                 align_corners=None):
        super().__init__()
        self.size = size
        if isinstance(scale_factor, tuple):
            self.scale_factor = tuple(float(factor) for factor in scale_factor)
        else:
            self.scale_factor = float(scale_factor) if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        if not self.size:
            size = [int(t * self.scale_factor) for t in x.shape[-2:]]
        else:
            size = self.size
        return resize(x, size, None, self.mode, self.align_corners)
