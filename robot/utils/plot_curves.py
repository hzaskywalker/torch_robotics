import numpy as np
from itertools import product
import matplotlib.pyplot as plt

linestyles = ['-', '--', '-.', ':']
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']


def plot_curve_with_variance(fig, mean, std=None, ylim=None, X=None, train_sizes=None,
                             label='Training score', color="r", linestyle='-'):
    """
    ylim (ymin, ymax)
    """
    mean = np.array(mean)
    if train_sizes is None:
        train_sizes = np.linspace(0, 1, num=mean.shape[0])

    if ylim is not None:
        plt.ylim(ylim)

    if std is not None:
        plt.fill_between(train_sizes, mean - std,
                         mean + std, alpha=0.1,
                         color=color)
    plt.plot(train_sizes, mean, '-', color=color,
             label=label, linestyle=linestyle)
    plt.legend(loc="best")
    return fig


def plot_curves(field, ylim=None):
    assert len(
        field) < 32, "current we only suppor the plot for 32 lines, we will add node marker later"
    fig = plt.figure()
    num = len(field)

    for (key, val), (s, c) in zip(field.items(), product(linestyles, colors)):
        fig = plot_curve_with_variance(
            fig, val, label=key, color=c, linestyle=s, ylim=ylim)
    plt.show()


def plot_count(vals, show=True):
    count = {}
    for l in vals:
        l = int(l)
        if l not in count:
            count[l] = 0
        count[l] += 1
    print(count)
    if not show:
        return
    x = sorted([i for i in count.keys()])
    y = [count[i] for i in x]
    from tools.utils import plot_curves
    from tools import Field
    plot_curves(Field(xx=y))