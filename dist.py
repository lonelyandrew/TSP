import numpy as np


def cal_dist():
    data = np.loadtxt('ja9847.txt', skiprows=7, usecols=(1, 2))
    np.save('ja9847.npy', data)
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        diff = data[i] - data
        diff = np.power(diff, 2)
        dist_matrix[:, i] = np.sum(diff, axis=1)
    dist_matrix = np.sqrt(dist_matrix)
    np.save('dist', dist_matrix)


if __name__ == '__main__':
    cal_dist()
