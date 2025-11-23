"""Minimal numpy stub for offline environments."""
import math
import random as _random


def sqrt(x):
    return math.sqrt(x)


class RandomState:
    def __init__(self, seed=None):
        self._random = _random.Random(seed)

    def normal(self, loc, scale, size):
        return [self._random.gauss(loc, scale) for _ in range(size)]


class _RandomModule:
    RandomState = RandomState


random = _RandomModule()
