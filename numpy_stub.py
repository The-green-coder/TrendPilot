"""Minimal numpy stub for offline environments."""
import math
import random as _random


def sqrt(x):
    return math.sqrt(x)


def isnan(x):
    try:
        return math.isnan(x)
    except Exception:
        return False


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


class RandomState:
    def __init__(self, seed=None):
        self._random = _random.Random(seed)

    def normal(self, loc, scale, size):
        return [self._random.gauss(loc, scale) for _ in range(size)]


class _RandomModule:
    RandomState = RandomState


random = _RandomModule()


__all__ = [
    "sqrt",
    "isnan",
    "isclose",
    "RandomState",
    "random",
]
