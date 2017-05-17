#!/usr/bin/env python3

import numpy as np
import logging


class TSP:
    '''An implementaion of MO-GA to solve TSP problem.
    '''

    def __init__(self, turns):
        '''Init with data and several hyper-parameters.

        Args:
            turns: How many turns you wanna perform before stopping.
        '''
        self.config_logger()
        logging.info('*' * 120)
        logging.info('-*- TSP INITIALIZATION STARTED -*-')
        self.data = np.loadtxt('ja9847.txt', skiprows=7, usecols=(1, 2))
        self.dist = np.load('dist.npy')
        self.m = len(self.data)
        self.n = 100
        self.pm = 0.3
        self.q = 50
        self.turns = turns
        self.population = np.zeros([self.n, self.m], dtype=int)
        self.best_min_dist = float('inf')
        self.best_solution = None
        self.best_turn = -1
        logging.info('CITIES:%d', self.m)
        logging.info('POPULATION:%d', self.n)
        logging.info('PERMUTATION PROBABILITY:%.2f', self.pm)
        logging.info('ELITE COUNT:%d', self.q)
        logging.info('TURNS:%d', self.turns)
        logging.info('-*- TSP INITIALIZATION FINISHED -*-\n')

    def config_logger(self):
        '''Config the logger.
        '''
        logging.basicConfig(filename='tsp.log',
                            level=logging.DEBUG,
                            format='%(asctime)s-%(levelname)s-%(message)s')
        formatter = logging.Formatter(
            '%(asctime)s-%(levelname)s-%(message)s')
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

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
        self.population = np.apply_along_axis(self.align, 1, self.population)
        self.population = self.sort(self.population)

    def evolve(self):
        '''Evolve the population.
        '''
        logging.info('-*- EVOLUTION STARTED -*- ')
        for turn in range(self.turns):
            offsprings = np.array([self.crossover(p1, p2)
                                   for p1, p2 in self.pair_parents()])
            offsprings = np.reshape(offsprings, (4 * len(offsprings), self.m))
            offsprings = self.mutation(offsprings)
            self.select(offsprings)
            min_dist, min_index = self.find_min()
            if self.best_min_dist > min_dist:
                self.best_min_dist = min_dist
                self.best_solution = self.population[min_index]
                self.best_turn = turn
            logging.info('TURN %d - AVG: %.3f - MIN: %.3f',
                         turn, self.avg_dist(), min_dist)
        logging.info('-' * 120)
        logging.info('MIN DIST: %.3f IN TURN %d',
                     self.best_min_dist, self.best_turn)
        logging.info('-*- EVOLUTION FINISHED -*- ')
        logging.info('*' * 120)

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
    tsp = TSP(5)
    tsp.init_population()
    tsp.evolve()
