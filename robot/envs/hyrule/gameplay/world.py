# chemistry
# We use
from .object import Object
from collections import OrderedDict
from .instruction import Magic, Cost


class World(Object):
    # World is a special object that handles everything
    # Gameplay stores the current representation of the environment
    # It will translate the input commend and the modification of the state into the real actions in the world..

    # In another word, it's a parser that parse the high level command into low-level actions.

    # simulator should support add action, change the position of the object

    # one can compare parser with torch.optim
    def __init__(self, simulator, objects, optimizer=None):
        """
        :param simulator: simulator, the machine..
        :param objects: a dict of object...
        :param optimizer: the optimizer that we would use to optimize the actions, current it must be None
        """
        super(World, self).__init__(None, None)
        self.objects = OrderedDict()

        for name, obj in objects.items():
            # those objects are pysapien_objects
            if not isinstance(obj, Object):
                obj = Object(obj, self)
            obj.linkto(self)

            self.objects[name] = obj

        self.optimizer = optimizer
        self.sim = simulator

        self._relation_set = {}

    def step(self):
        self.forward(self.sim) # set all constraints..
        self.sim.step() # step for one step...
        # TODO: we should explicity distinguish actions,
        self.backward(self.sim)
        return self

    def optimize(self, instructions, costs):
        if len(costs) > 0:
            if self.optimizer is None:
                raise NotImplementedError("optimizer is None")
            instructions = self.optimizer(self.sim, instructions, costs)
        return instructions

    def rollout(self, horizon):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError


    def register(self, name, type):
        self._relation_set[name] = type
        return self

    def __getattr__(self, item):
        # TODO: very strange function...
        # The main difference between __getattr__ and __getattribute__ is that if the attribute was not found by the usual way then __getattr__ is used.
        if item in self.objects:
            return self.objects[item]
        elif item in self._relation_set:
            out = self._relation_set[item]
            def run(obj, *args, parent=None):
                obj.add_relation(out(*args), parent)
                return self
            return run
        else:
            raise AttributeError

    def render(self, sleep=0):
        self.sim.render()
        if sleep > 0:
            import time
            time.sleep(sleep)
        return self
