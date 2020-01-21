from .dict import Dict
from .angular import Angular6d
from .array import Array, Discrete

def Node(n, m=None, low=1., high=None):
    if m is None:
        return Dict(
            p=Array(low=low, high=high, shape=(n, 3)),
            v=Array(low=low, high=high, shape=(n, 3)),
            w=Angular6d(low=-1., high=None, shape=(n, 6)),
            dw=Array(low=low, high=high, shape=(n, 3)),
        )
    else:
        return Array(
            low=low, high=high, shape=(n, m)
        )


def GraphSpace(n, m, dim_edge, dim_node=None,
               low_node=1, high_node=None,
               low_edge=1, high_edge=None,
               global_dim=None):
    space = Dict()
    if dim_node is None or (dim_node is not None and dim_node != 0):
        space['node']= Node(n, dim_node, low_node, high_node)
    if dim_edge != 0:
        space['edge'] = Array(low_edge, high_edge, shape=(m, dim_edge))

    space['graph'] = Discrete(n, shape=(2, m))

    if global_dim is not None:
        space['global'] = Array(1, shape=(global_dim,))
    return space
