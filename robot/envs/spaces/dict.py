# dict that can do serialization
# copied from openai_gym
from collections import OrderedDict
from .space import Space


class Dict(Space):
    def __init__(self, spaces=None, **spaces_kwargs):
        assert (spaces is None) or (not spaces_kwargs), 'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces = spaces
        for space in spaces.values():
            assert isinstance(space, Space), 'Values of the dict should be instances of gym.Space'
        super(Dict, self).__init__(None, None) # None for shape and dtype, since it'll require special handling

    def seed(self, seed=None):
        [space.seed(seed) for space in self.spaces.values()]

    def sample(self):
        return OrderedDict([(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __getitem__(self, key):
        return self.spaces[key]

    def __repr__(self):
        return "Dict(" + ", ". join([str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def __eq__(self, other):
        return isinstance(other, Dict) and self.spaces == other.spaces