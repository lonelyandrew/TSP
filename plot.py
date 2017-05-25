#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.axes import Axes
# from matplotlib.lines import Line2D


def plot_sln_dym(sln):
    plt.axis([120000, 150000, 20000, 50000])
    plt.ion()
    data = np.load('ja9847.npy')
    for i in sln:
        plt.scatter(data[i, 1], data[i, 0], marker=',', s=1)
        # l = Line2D([data[i-1, 1], data[i-1, 0]], [data[i, 1], data[i, 0]])
        plt.pause(0.00005)
    while True:
        plt.pause(0.05)


def plot_sln(sln):
    pass


if __name__ == '__main__':
    e = np.load('init_int_200.npy')
    plot_sln_dym(e[0])
