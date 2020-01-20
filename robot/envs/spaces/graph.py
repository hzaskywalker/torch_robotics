from .space import Type, Space
from .dict import DictSpace, Dict
from .angular import Angular6d, Angular6dSpace, Array


class NodeSpace(DictSpace):
    def __init__(self, n):
        super(NodeSpace, self).__init__(
            node = Ar
        )


class Graph(DictSpace):
    pass
