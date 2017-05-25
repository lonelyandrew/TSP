#!/usr/bin/env python3

import logging
from datetime import datetime

import numpy as np


class Preprocessing:

    def __init__(self):
        self.dist = np.load('dist.npy')
        self.m = len(self.dist)
        self.config_logger()

    def find_elites(self, n):
        group = np.random.choice(list(range(self.m)), n, replace=False)
        self.log('-' * 120)
        self.log('PREPROCESSING: FIND {0} ELITES START'.format(n))
        start = datetime.now()
        init_elites = np.zeros((n, self.m), dtype=int)
        for i in range(n):
            source = group[i]
            sln = [source]
            whole = list(range(self.m))
            whole.remove(source)
            for _ in range(self.m - 1):
                src_city = sln[-1]
                next_city = whole[np.argmin(self.dist[src_city, whole])]
                sln.append(next_city)
                whole.remove(next_city)
            log_str = '({0}/{1}) START FROM {2}:{3}'
            self.log(log_str.format(i+1, n, source, self.path_dist(sln)))
            init_elites[i] = sln
        np.save('init_{}'.format(n), init_elites)
        end = datetime.now()
        self.log('TIME SPENDING: {0}'.format(end - start))
        self.log('PREPROCESSING: FIND {0} ELITES FINISHED'.format(n))

    def find_all_elites(self):
        self.log('-' * 120)
        self.log('PREPROCESSING: FIND ALL ELITES START')
        start = datetime.now()
        init_elites = np.zeros((self.m, self.m))
        for i in range(self.m):
            sln = [i]
            whole = list(range(self.m))
            whole.remove(i)
            for _ in range(self.m - 1):
                src_city = sln[-1]
                next_city = whole[np.argmin(self.dist[src_city, whole])]
                sln.append(next_city)
                whole.remove(next_city)
            self.log('START FROM {0}:{1}'.format(i, self.path_dist(sln)))
            init_elites[i] = sln
        np.save('init', init_elites)
        end = datetime.now()
        self.log('TIME SPENDING: {0}'.format(end - start))
        self.log('PREPROCESSING: FIND ALL ELITES FINISHED')

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

    def path_dist(self, sln):
        dist_sum = 0
        for i in range(self.m - 1):
            dist_sum += self.dist[sln[i], sln[i + 1]]
        else:
            dist_sum += self.dist[sln[-1]][sln[0]]
        return dist_sum


if __name__ == '__main__':
    pre = Preprocessing()
    pre.find_elites(200)
    # e = np.load('init_200.npy')
    # print(e)
