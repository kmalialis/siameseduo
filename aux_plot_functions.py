# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

###########
# 2D plot #
###########


def plot1d(data1):
    colours = ['pink', 'brown', 'orange', 'navy', 'olive', 'green', 'cyan', 'purple', 'grey', 'red', 'black', 'magenta']
    num_classes = len(np.unique(data1[:, -1]))

    fig, ax = plt.subplots()
    for c in range(num_classes):
        data_c = data1[data1[:, -1] == c]
        x = data_c[:, 0]
        ax.scatter(x, np.zeros(x.shape[0]), c=colours[c], s=5, alpha=1.0)

    ax.legend()
    ax.grid(True)
    plt.grid(linestyle='dotted')

    plt.xlabel('X1', fontsize=12, weight='bold')
    # plt.xlim(-0.5, 1.5)
    # plt.xticks([-0.5, 0.0, 0.5, 1.0, 1.5], fontsize=12)

    plt.ylabel('X2', fontsize=12, weight='bold')
    plt.ylim(-0.5, 0.5)

    plt.show()
    # fig.savefig(out_dir + 'test.pdf', bbox_inches='tight')


def plot2d(data1):
    colours = ['pink', 'brown', 'orange', 'navy', 'olive', 'green', 'cyan', 'purple', 'grey', 'red', 'black', 'magenta']
    num_classes = len(np.unique(data1[:, -1]))

    fig, ax = plt.subplots()
    for c in range(num_classes):
        data_c = data1[data1[:, -1] == c]
        x = data_c[:, 0]
        y = data_c[:, 1]
        ax.scatter(x, y, c=colours[c], s=5, alpha=1.0)

    ax.legend()
    ax.grid(True)
    plt.grid(linestyle='dotted')

    plt.xlabel('X1', fontsize=12, weight='bold')
    # plt.xlim(0.0, 1.0)
    # plt.xticks([-0.5, 0.0, 0.5, 1.0, 1.5], fontsize=12)

    plt.ylabel('X2', fontsize=12, weight='bold')
    # plt.ylim(0.0, 1.0)
    # plt.yticks([-0.5, 0.0, 0.5, 1.0, 1.5], fontsize=12)

    plt.show()
    # fig.savefig(out_dir + 'test.pdf', bbox_inches='tight')

###########
# 3D plot #
###########


def plot3d(data1):
    colours = ['pink', 'brown', 'orange', 'navy', 'olive', 'green', 'cyan', 'purple', 'grey', 'red', 'black', 'magenta']
    num_classes = len(np.unique(data1[:, -1]))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for c in range(num_classes):
        data_c = data1[data1[:, -1] == c]
        x = data_c[:, 0]
        y = data_c[:, 1]
        z = data_c[:, 2]
        ax.scatter(x, y, z, c=colours[c])

    ax.legend()
    ax.grid(True)
    plt.show()
