# -*- coding: utf-8 -*-
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')


def create_plots(list_filenames, list_legend_names, loc='lower right'):
    size = 42
    size_legend = 25
    fig = plt.figure(figsize=(12, 12))

    for i in range(len(list_filenames)):
        arr = np.loadtxt(list_filenames[i], delimiter=', ')         # load data

        print(arr.shape)

        means = np.mean(arr, axis=0)                                # y-axis values
        x_axis = np.arange(means.shape[0])                          # x-axis values
        se = np.std(arr, ddof=1, axis=0) / np.sqrt(arr.shape[0])    # standard error (ddof=1 for sample)

        plt.plot(x_axis, means, label=str(list_legend_names[i]), linewidth=3.0)
        plt.fill_between(x_axis, means - se, means + se, alpha=0.2)

    # x-axis
    plt.xlim(0, arr.shape[1])

    plt.xlabel('Time Step', fontsize=size, weight='bold')
    plt.xticks(fontsize=size)
    plt.xticks([0.0, 6000, 12000, 18000], fontsize=size)  # gestures

    # y-axis
    plt.ylabel('G-mean', fontsize=size, weight='bold')
    plt.yticks(np.arange(0.0, 1.000001, 0.2), fontsize=size)
    plt.ylim(0.0, 1.0)

    # legend
    if 1:
        leg = plt.legend(ncol=1, loc=loc, fontsize=size_legend)
        leg.get_frame().set_alpha(0.9)

    # grid
    plt.grid(linestyle='dotted')

    # plot
    plt.show()

    # save
    # fig.savefig(out_dir + 'test.png', bbox_inches='tight')


out_dir = "exps/"
filenames = [
    out_dir + "sea_abrupt_actisiamese10_0.01_DA5['interpolation', 'extrapolation', 'gaussian_noise']_preq_gmean.txt"
]
create_plots(filenames, ['SiameseDuo++'])