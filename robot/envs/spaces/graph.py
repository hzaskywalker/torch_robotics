import numpy as np
from .space import Type, Space
from .dict import DictSpace, Dict
from .angular import Angular6d, Angular6dSpace
from .array import ArraySpace, Discrete


def NodeSpace(n, m=None, low=1., high=None):
    if m is None:
        return DictSpace(
            p=ArraySpace(low=low, high=high, shape=(n, 3)),
            v=ArraySpace(low=low, high=high, shape=(n, 3)),
            w=Angular6dSpace(shape=(n,)),
            dw=ArraySpace(low=low, high=high, shape=(n, 3)),
        )
    else:
        return ArraySpace(
            low=low, high=high, shape=(n, m)
        )


def GraphSpace(n, m, dim_edge, dim_node=None,
               low_node=1, high_node=None,
               low_edge=1, high_edge=None,
               global_dim=None):
    space = DictSpace()
    if dim_node is not None and dim_node != 0:
        space['node']=NodeSpace(n, dim_node, low_node, high_node),
    if dim_edge != 0:
        space['edge'] = ArraySpace(low_edge, high_edge, shape=(m, dim_edge))

    space['graph'] = Discrete(n, shape=(2, m))

    if global_dim is not None:
        space['global'] = ArraySpace(1, shape=(global_dim,))
    return space
