#!/usr/bin/env python3

from datetime import datetime
import logging

import itchat
import numpy as np


class TSTP:

    def __init__(self,
                 generations,
                 population,
                 elite_p,
                 p_m,
                 init_p,
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
        self.population_init(init_p)
        self.log('-*- TSP INITIALIZATION FINISHED -*-')
        self.log('*' * 120)

    def population_init(self, proportion):
        added_sln = np.load('init_int_200.npy')
        added_sln_count = round(self.n * proportion)
        self.population[:added_sln_count] = added_sln[np.random.choice(
            added_sln.shape[0], added_sln_count, replace=False)]
        for i in range(self.n - added_sln_count):
            self.population[added_sln_count + i,
                            :] = np.random.permutation(self.m)
        self.sort(self.population)
        self.log(
            'INIT FINISHED: {0} / {1} ADDED'.format(added_sln_count, self.n))
        self.log('INIT AVG DIST: {}'.format(self.avg_dist()))
        self.log('INIT MIN DIST: {}'.format(
            self.path_dist(self.population[0])))

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
            itchat.send('{0}-{1}'.format(datetime.now(),
                                         message), toUserName='filehelper')

    def sort(self, population):
        dist_list = np.array([self.path_dist(v)
                              for v in population])
        return population[dist_list.argsort()]

    def path_dist(self, sln):
        dist_sum = 0
        for i in range(self.m - 1):
            dist_sum += self.dist[sln[i], sln[i + 1]]
        else:
            dist_sum += self.dist[sln[-1], sln[0]]
        return dist_sum

    def avg_dist(self):
        dist_list = np.array([self.path_dist(v) for v in self.population])
        return np.average(dist_list)

    def evolve(self):
        self.log('-*- EVOLUTION STARTED -*- ')
        for generation in range(self.generations):
            pts = self.choose_crossover_pt()
            self.log(
                'GENERATION {0} - PTS {1[0]} & {1[1]}'.format(generation, pts))
            offsprings = []
            for p1, p2 in self.pair_parents():
                new_offsprings = self.crossover(p1, p2, pts)
                offsprings += new_offsprings
            offsprings = np.array(offsprings)
            self.log('GENERATION {0} - CROSSOVER DONE'.format(generation))
            offsprings = self.mutation(offsprings)
            self.log('GENERATION {0} - MUTATION DONE'.format(generation))
            self.select(offsprings)
            self.log('GENERATION {0} - SELECT DONE'.format(generation))
            min_dist = self.path_dist(self.population[0])
            self.log('GENERATION {0:04d} - AVG: {1:.3f} - MIN: {2:.3f}'.format(
                     generation, self.avg_dist(), min_dist))
            self.update_best_solution(generation)
        self.log('-' * 120)
        self.log('MIN DIST: {0:.3f} IN GENERATION {1}'.format(
                 self.best_min_dist, self.best_generation))
        self.log('-*- EVOLUTION FINISHED -*- ')
        self.log('*' * 120)

    def update_best_solution(self, generation):
        np.save('min_gen{0}'.format(generation), self.population[0])
        min_dist = self.path_dist(self.population[0])
        if min_dist < self.best_min_dist:
            self.best_min_dist = min_dist
            self.best_generation = generation
            self.best_solution = self.population[0]

    def select(self, offsprings):
        selecting = self.population[0:self.q]
        selecting = np.vstack((selecting, offsprings))
        selecting = self.sort(selecting)
        self.population = selecting[:self.n]

    def fitness(self, length):
        beta = 0.2
        fitness_list = np.arange(length, dtype=float)
        fitness_list[::2] = np.apply_along_axis(
            lambda i: (length - i + 1) / length, 0, fitness_list[::2])
        fitness_list[1::2] = np.apply_along_axis(
            lambda i: beta * np.power(1 - beta, i - 1), 0, fitness_list[1::2])
        return fitness_list

    def choose_crossover_pt(self):
        pt1, pt2 = (0, 0)
        while pt1 == pt2:
            pt1, pt2 = sorted(np.random.randint(1, self.m - 1, size=2))
        return pt1, pt2

    def pair_parents(self):
        fits = self.fitness(self.n)
        sum_fits = sum(fits)
        p = [f / sum_fits for f in fits]
        cul_p = np.array([sum(p[:i]) for i in range(self.n)])
        for i in range(0, self.n, 2):
            prob1, prob2 = np.random.random_sample(2)
            yield np.argmax(cul_p > prob1), np.argmax(cul_p > prob2)

    def crossover(self, x_i, y_i, pts):
        pt1, pt2 = pts
        x, y = self.population[x_i], self.population[y_i]
        c1, c2 = x[:], y[:]
        c1[:pt1] = x[np.argsort(y[:pt1])]
        c2[:pt1] = y[np.argsort(x[:pt1])]
        c1[pt1:pt2] = x[np.argsort(y[pt1:pt2])+pt1]
        c2[pt1:pt2] = y[np.argsort(x[pt1:pt2])+pt1]
        c1[pt2:] = x[np.argsort(y[pt2:])+pt2]
        c2[pt2:] = y[np.argsort(x[pt2:])+pt2]
        return c1, c2

    def mutation(self, offsprings):
        m_count = round(self.pm * len(offsprings))
        m_offsprings = np.random.choice(np.arange(len(offsprings)), m_count)
        pt1, pt2 = (0, 0)
        while pt1 == pt2:
            pt1, pt2 = sorted(np.random.randint(1, self.m - 2, size=2))
        for i in m_offsprings:
            offsprings[i][pt1:pt2] = offsprings[i][pt2 - 1:pt1 - 1:-1]
        return offsprings


if __name__ == '__main__':
    tstp = TSTP(generations=20,
                population=500,
                elite_p=0.8,
                p_m=0.1,
                init_p=0.4)
    tstp.evolve()
