#!/usr/bin/env python3

import numpy as np
import logging
import itchat
import os
import datetime


class TSP:
    '''An implementaion of MO-GA to solve TSP problem.
    '''

    def __init__(self, generations, population):
        '''Init with data and several hyper-parameters.

        Args:
            generations: How many generations you wanna evolve.
            population: The size of the population.
        '''
        self.config_itchat()
        self.config_logger()
        self.log('*' * 120)
        self.log('-*- TSP INITIALIZATION STARTED -*-')
        self.data = np.loadtxt('ja9847.txt', skiprows=7, usecols=(1, 2))
        self.dist = np.load('dist.npy')
        self.m = len(self.data)
        self.n = population
        self.pm = 0.1
        self.q = round(population / 2)
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
        self.init_population()
        self.log('-*- TSP INITIALIZATION FINISHED -*-\n')

    def config_itchat(self):
        if __debug__:
            itchat.auto_login(enableCmdQR=2, hotReload=True)
        else:
            itchat.auto_login(hotReload=True)

    def log(self, message):
        logging.info(message)
        itchat.send('{0}-{1}'.format(datetime.datetime.now(),
                                     message), toUserName='filehelper')

    def config_logger(self):
        '''Config the logger.
        '''
        logging.basicConfig(filename='tsp.log',
                            level=logging.INFO,
                            format='%(asctime)s-%(levelname)s-%(message)s')
        formatter = logging.Formatter(
            '%(asctime)s-%(levelname)s-%(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    def save_best_solution(self):
        if os.path.isfile('min_dist.txt'):
            min_dist = np.loadtxt('min_dist.txt')
            if min_dist > self.best_min_dist:
                np.savetxt('min_dist.txt', [self.best_min_dist], '%.3f')
                np.save('min.npy', self.best_solution)
                self.log('HISTORY MIN VALUE REFRESHED TO {:.3f}'.format(
                    self.best_min_dist))
        else:
            np.savetxt('min_dist.txt', [self.best_min_dist], '%.3f')
            np.save('min.npy', self.best_solution)
            self.log('HISTORY MIN VALUE REFRESHED TO {:.3f}'.format(
                self.best_min_dist))

    def align(self, x):
        '''Align a individual to make it start from 1.

        Args:
            x: The individual that will be shifted.
        '''
        i, = np.where(x == 0)
        return np.roll(x, -i)

    def init_population(self):
        '''Initialize the population.
        '''
        for i in range(self.n):
            self.population[i, :] = np.random.permutation(self.m)
        # self.population = np.apply_along_axis(self.align, 1, self.population)
        if os.path.isfile('min.npy') and os.path.isfile('min_dist.txt'):
            history_min = np.load('min.npy')
            history_min_dist = np.loadtxt('min_dist.txt')
            replace_count = round(0.3 * self.n)
            self.population[:replace_count] = history_min
            self.log('INHERIT {0} INDIVIDUALS WITH {1}'.format(
                replace_count, history_min_dist))
        else:
            self.log('INIT FROM PURE RANDOM STATE')

        self.population = self.sort(self.population)

    def evolve(self):
        '''Evolve the population.
        '''
        self.log('-*- EVOLUTION STARTED -*- ')
        for generation in range(self.generations):
            offsprings = np.array([self.crossover(p1, p2)
                                   for p1, p2 in self.pair_parents()])
            offsprings = np.reshape(offsprings, (4 * len(offsprings), self.m))
            offsprings = self.mutation(offsprings)
            self.select(offsprings)
            min_dist, min_index = self.find_min()
            if self.best_min_dist > min_dist:
                self.best_min_dist = min_dist
                self.best_solution = self.population[min_index]
                self.best_generation = generation
            self.log('GENERATION {0:04d} - AVG: {1:.3f} - MIN: {2:.3f}'.format(
                     generation, self.avg_dist(), min_dist))
            if generation % 50 == 1:
                self.save_best_solution()
        self.log('-' * 120)
        self.log('MIN DIST: {0:.3f} IN GENERATION {1}'.format(
                 self.best_min_dist, self.best_generation))
        self.save_best_solution()
        self.log('-*- EVOLUTION FINISHED -*- ')
        self.log('*' * 120)

    def find_min(self):
        '''Find the minimum path distance and its index among current
           population.
        '''
        dist_list = np.array([self.path_dist(v) for v in self.population])
        return np.min(dist_list), np.argmin(dist_list)

    def avg_dist(self):
        '''Calculate the average path distance of the current
           population.

        Returns:
            Return the tuple (avg, min).
        '''
        dist_list = np.array([self.path_dist(v) for v in self.population])
        return np.average(dist_list)

    def select(self, offsprings):
        '''Select next generation of population among current q elitest
           individuals and 2n offsprings.

        Args:
            offsprings: The offsprings of current generation.
        '''
        selecting = self.population[0:self.q]
        selecting = np.vstack((selecting, offsprings))
        selecting = self.sort(selecting)
        fits = self.fitness(len(selecting))
        sum_fits = sum(fits)
        p = [f / sum_fits for f in fits]
        cul_p = np.array([sum(p[:i]) for i in range(len(selecting))])
        new_population = []
        for i in range(self.n):
            prob = np.random.random_sample()
            choice = np.argmax(cul_p > prob)
            new_population.append(selecting[choice])
        self.population = new_population

    def sort(self, population):
        '''Sort the whole population depending on path distance.

        Args:
            population: The population which is going to be sorted.
        '''
        dist_list = np.array([self.path_dist(v) for v in population])
        population = population[dist_list.argsort()]
        return population

    def fitness(self, length):
        '''Calculate the fitness value of a individual.

        Args:
            length: The length of the population.

        Returns:
            Return a list of fitness values.
        '''
        beta = 0.2
        fitness_list = np.arange(length, dtype=float)
        fitness_list[::2] = np.apply_along_axis(
            lambda i: (length - i + 1) / length, 0, fitness_list[::2])
        fitness_list[1::2] = np.apply_along_axis(
            lambda i: beta * np.power(1 - beta, i - 1), 0, fitness_list[1::2])
        return fitness_list

    def path_dist(self, v):
        '''Calculate the whole distance of a individual.

        Args:
            v: An individual in the population.

        Returns:
            Return a summation of the whole distances
        '''
        dist_sum = 0
        for i in range(self.m - 1):
            dist_sum += self.dist[v[i], v[i + 1]]
        return dist_sum

    def pair_parents(self):
        '''Select pairs of parents among the population.

        Yields:
            Yields a list whose elements are tuples of parents' index (p1, p2).
        '''
        fits = self.fitness(self.n)
        sum_fits = sum(fits)
        p = [f / sum_fits for f in fits]
        cul_p = np.array([sum(p[:i]) for i in range(self.n)])
        for i in range(0, self.n, 2):
            prob1, prob2 = np.random.random_sample(2)
            yield np.argmax(cul_p > prob1), np.argmax(cul_p > prob2)

    def crossover(self, x_i, y_i):
        '''Perform a crossover operation on two parents to produce four offsprings.

        Args:
            x_i: The first parent's index.
            y_i: The second parent's index.

        Returns:
            Return a list consists of four generated offsprings.
        '''
        x, y = self.population[x_i], self.population[y_i]
        pt1, pt2 = (0, 0)
        while pt1 == pt2:
            pt1, pt2 = sorted(np.random.randint(1, self.m - 1, size=2))
        k = self.m - pt2
        c1, c2 = np.roll(x, k), np.roll(y, k)
        middle_1, middle_2 = x[pt1:pt2], y[pt1:pt2]
        c1_prime = np.array([i for i in c1 if i not in middle_2])
        c2_prime = np.array([i for i in c2 if i not in middle_1])
        c1 = np.concatenate((c1_prime[k:], middle_2, c1_prime[:k]))
        c2 = np.concatenate((c2_prime[k:], middle_1, c2_prime[:k]))
        c3, c4 = np.array(x), np.array(y)
        c3[:pt2 - pt1], c4[:pt2 - pt1] = middle_1, middle_2
        c3[pt2 - pt1:pt2], c4[pt2 - pt1:pt2] = x[:pt1], y[:pt1]
        section3_x, section3_y = c3[pt2:], c4[pt2:]
        c3 = np.array([i for i in c3 if i not in section3_y])
        c4 = np.array([i for i in c4 if i not in section3_x])
        c3 = np.concatenate((c3, section3_y))
        c4 = np.concatenate((c4, section3_x))
        return (c1, c2, c3, c4)

    def mutation(self, offsprings):
        '''Perform a mutation operation on a individual of new offsprings.

        Args:
            offsprings: The offsprings those are going to be mutated.
        '''
        m_count = round(self.pm * len(offsprings))
        m_offsprings = np.random.choice(np.arange(len(offsprings)), m_count)
        for o in offsprings[m_offsprings]:
            pt1, pt2 = (0, 0)
            while pt1 == pt2:
                pt1, pt2 = sorted(np.random.randint(1, self.m - 1, size=2))
            o[pt1:pt2] = o[pt2 - 1:pt1 - 1:-1]
        return offsprings

    def cal_dist():
        '''Calculate all distances between every pair of cities.
        And save the result matrix to the file.
        '''
        data = np.loadtxt('ja9847.txt', skiprows=7, usecols=(1, 2))
        n = data.shape[0]
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            diff = data[i] - data
            diff = np.power(diff, 2)
            dist_matrix[:, i] = np.sum(diff, axis=1)
        dist_matrix = np.sqrt(dist_matrix)
        dist_matrix = np.round(dist_matrix, 3)
        np.save('dist', dist_matrix)


if __name__ == '__main__':
    if __debug__:
        tsp = TSP(generations=5, population=10)
        tsp.evolve()
    else:
        tsp = TSP(generations=5000, population=100)
        tsp.evolve()
