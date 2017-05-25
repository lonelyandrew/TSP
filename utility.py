#!/usr/bin/env python3

import numpy as np


def verify(sln):
    for i, _ in enumerate(sln):
        if i not in sln:
            return False
    return True


if __name__ == '__main__':
    e = np.load('min_gen1.npy')
    print(verify(e))
