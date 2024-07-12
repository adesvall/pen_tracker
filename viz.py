import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from more_itertools import unzip

def plot_warping_path(paths, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.yaxis.set_inverted(True)
    ax.set_aspect('equal', 'box')
    for i, p in enumerate(paths):
        x, y = unzip(p)
        if labels:
            plt.plot(list(x), list(y), label=labels[i])
        else:
            plt.plot(list(x), list(y))
    plt.legend()
    plt.show()

def plot_trajectory(ts, ax : plt.Axes, colors=None):
    if colors is None:
        colors = plt.cm.jet(np.linspace(0, 1, len(ts)))
    for i in range(len(ts) - 1):
        ax.plot(ts[i:i+2, 0], ts[i:i+2, 1],
                marker='o', c=colors[i%len(colors)], markersize=1)
    ax.set_aspect('equal')

def plot_trajectories(curves, bary, ax=None):
    if ax is None:
        fig = plt.gcf()
        ax = fig.add_subplot()
    ax.set_aspect('equal', 'box')
    for curve in curves:
        plot_trajectory(curve, ax)
    plot_trajectory(bary, ax, colors=['black'])
    return ax

        
def plot_2_trajectories(ts0, ts1, path=None, fname_figure=None):
    ax0 = plt.subplot(1, 2, 1)
    plot_trajectory(ts0, ax0)
    plt.xticks([])
    plt.yticks([])

    ax1 = plt.subplot(1, 2, 2)
    plot_trajectory(ts1, ax1)
    plt.xticks([])
    plt.yticks([])

    if path:
        for i, j in path:
            xyA = ts0[i]
            xyB = ts1[j]
            con1 = ConnectionPatch(xyA=xyA, xyB=xyB, 
                                coordsA="data", coordsB="data", 
                                axesA=ax0, axesB=ax1, 
                                color="orange")
            ax1.add_artist(con1)

    plt.tight_layout()
    if fname_figure:
        plt.savefig(fname_figure)
    else:
        plt.show()


def plot_predictions(inputs, outputs, preds):
    plt.title('Input, Output, Prediction')
    k = 5
    for i in range(k):
        plt.subplot(2, k, i + 1)
        plt.gca().set_title(f"Ground Truth {i}")
        plt.gca().set_aspect('equal')
        # plot_trajectory(inputs[i], plt.gca())
        plot_trajectory(outputs[i], plt.gca())
        plt.subplot(2, k, k + i + 1)
        plt.gca().set_title(f"Prediction {i}")
        plt.gca().set_aspect('equal')
        plot_trajectory(preds[i], plt.gca(), ['black'])
    plt.show()

