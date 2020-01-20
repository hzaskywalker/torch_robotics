import numpy as np
from .space import Type, Space
from .dict import DictSpace, Dict
from .angular import Angular6d, Angular6dSpace
from .array import ArraySpace, Discrete


def node_space(n):
    return DictSpace(
        p=ArraySpace(1, shape=(n, 3)),
        v=ArraySpace(1, shape=(n, 3)),
        w=Angular6dSpace(shape=(n,)),
        dw=ArraySpace(1, shape=(n, 3)),
    )

def edge_space(n, m, d):
    return DictSpace(
        e=ArraySpace(1, shape=(m, d)),
        s=Discrete(n, shape=(m,)),
        t=Discrete(n, shape=(m,))
    )

def graph_space(node_space, edge_space, global_space=None):
    space = DictSpace(
        node=node_space,
        edge=edge_space,
    )

    if global_space is not None:
        space['global'] = global_space
    return space
