import cPickle
from numpy import *


def edges2centers(edges):
    return array([low + (high - low) / 2.0
                  for low, high in zip(edges[:-1], edges[1:])])


class Memoize:
    """
    Memoize with mutable arguments
    """

    def __init__(self, function):
        self.function = function
        self.memory = {}

    def __call__(self, *args, **kwargs):
        hash_str = cPickle.dumps(args) + cPickle.dumps(kwargs)
        if not hash_str in self.memory:
            self.memory[hash_str] = self.function(*args, **kwargs)
        return self.memory[hash_str]
