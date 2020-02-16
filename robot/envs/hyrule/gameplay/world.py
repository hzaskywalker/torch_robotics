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
    def __init__(self, simulator, objects, panel, optimizer=None):
        """
        :param simulator: simulator, the machine..
        :param objects: a dict of object...
        :param optimizer: the optimizer that we would use to optimize the actions, current it must be None
        """
        super(World, self).__init__(None, None)
        self.simulator = simulator
        self.panel = panel
        self.objects = OrderedDict()

        for name, obj in objects.items():
            # those objects are pysapien_objects
            if not isinstance(obj, Object):
                obj = Object(obj, self)
            obj.linkto(self)

            self.objects[name] = obj

        self.optimizer = optimizer
        self._panel = panel

    def step(self, reward=False):
        self.execute(self.panel) # set all constraints..
        self.panel.step() # step for one step...
        # TODO: we should explicity distinguish actions,
        return self.backward(self.panel)

    def optimize(self, instructions, costs):
        if len(costs) > 0:
            if self.optimizer is None:
                raise NotImplementedError("optimizer is None")
            instructions = self.optimizer(self.simulator, instructions, costs)
        return instructions

    def rollout(self, horizon):
        raise NotImplementedError

    def parameters(self):
        raise NotImplementedError

    def __getattr__(self, item):
        # TODO: very strange function...
        # The main difference between __getattr__ and __getattribute__ is that if the attribute was not found by the usual way then __getattr__ is used.
        if item in self.objects:
            return self.objects[item]
        else:
            raise AttributeError
