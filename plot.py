#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.axes import Axes
# from matplotlib.lines import Line2D


def plot_sln_dym(sln):
    plt.axis([120000, 150000, 20000, 50000])
    plt.ion()
    data = np.load('ja9847.npy')
    for i, city in enumerate(sln):
        city_current = data[city]
        # plt.scatter(city_current[1], city_current[0], marker=',', s=0.5)
        if i > 0:
            city_prev = data[sln[i - 1]]
            plt.plot([city_prev[1], city_current[1]],
                     [city_prev[0], city_current[0]],
                     lw=2)
        plt.pause(0.00005)
    while True:
        plt.pause(0.05)


def plot_sln(sln):
    data = np.load('ja9847.npy')
    y = []
    x = []
    for i in sln:
        y.append(data[i, 0])
        x.append(data[i, 1])
    plt.plot(x, y, lw=1)
    plt.show()


if __name__ == '__main__':
    e = np.load('init_int_200.npy')
    plot_sln(e[0])
