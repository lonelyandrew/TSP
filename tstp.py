#!/usr/bin/env python3

import numpy as np
import logging
import itchat
# import os
import datetime


class TSTP:

    def __init__(self,
                 generations,
                 population,
                 elite_p,
                 p_m,
                 wechat_log=False):
        self.wechat_log = wechat_log
        if wechat_log:
            self.config_itchat()
        self.config_logger()
        self.log('*' * 120)
        self.log('-*- TSP INITIALIZATION STARTED -*-')
        self.data = np.load('ja9847.npy')
        self.dist = np.load('dist.npy')
        self.m = len(self.data)
        self.n = population
        self.pm = p_m
        self.q = round(population * elite_p)
        self.generations = generations
        self.population = np.zeros([self.n, self.m], dtype=int)
        self.best_min_dist = float('inf')
        self.best_solution = None
        self.best_generation = -1
        if __debug__:
            self.log('MODE:DEBUG')
        self.log('CITIES:{}'.format(self.m))
        self.log('POPULATION:{}'.format(self.n))
        self.log('PERMUTATION PROBABILITY:{:.2f}'.format(self.pm))
        self.log('ELITE COUNT:{}'.format(self.q))
        self.log('GENERATION:{}'.format(self.generations))
        # self.init_population(0.4)
        self.log('-*- TSP INITIALIZATION FINISHED -*-')
        self.log('*' * 120)

    def config_itchat(self):
        if __debug__:
            itchat.auto_login(enableCmdQR=2, hotReload=True)
        else:
            itchat.auto_login(hotReload=True)

    def config_logger(self):
        logging.basicConfig(filename='tsp.log',
                            level=logging.INFO,
                            format='%(asctime)s-%(levelname)s-%(message)s')
        formatter = logging.Formatter(
            '%(asctime)s-%(levelname)s-%(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def log(self, message):
        logging.info(message)
        if self.wechat_log:
            itchat.send('{0}-{1}'.format(datetime.datetime.now(),
                                         message), toUserName='filehelper')

    def sort(self, population):
        population = np.array(population)
        dist_list = np.array([self.path_dist(v)
                              for v in population])
        return population[dist_list.argsort()]

    def path_dist(self, sln):
        dist_sum = 0
        for i in range(self.m-1):
            dist_sum += self.dist[sln[i], sln[i+1]]
        else:
            dist_sum += self.dist[sln[-1], sln[0]]
        return dist_sum

    @staticmethod
    def cal_dist():
        data = np.loadtxt('ja9847.txt', skiprows=7, usecols=(1, 2))
        np.save('ja9847.npy', data)
        # n = data.shape[0]
        # dist_matrix = np.zeros((n, n))
        # for i in range(n):
        #     diff = data[i] - data
        #     diff = np.power(diff, 2)
        #     dist_matrix[:, i] = np.sum(diff, axis=1)
        # dist_matrix = np.sqrt(dist_matrix)
        # np.save('dist', dist_matrix)


if __name__ == '__main__':
    TSTP.cal_dist()
