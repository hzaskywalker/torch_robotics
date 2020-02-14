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
        super(World, self).__init__(simulator, None)
        self.objects = OrderedDict()

        for name, obj in objects.items():
            # those objects are pysapien_objects
            if not isinstance(obj, Object):
                obj = Object(obj, self)
            obj.linkto(self)

            self.objects[name] = obj

        self.optimizer = optimizer
        self.objects = {}

    def parse(self):
        # how to parse it...
        instructions = super(World, self).parse()
        instructions = self.optimize(instructions)
        return instructions

    def optimize(self, instructions):
        need_optimize = False
        for i in instructions:
            need_optimize = need_optimize or  isinstance(i, Cost)
        if need_optimize:
            if self.optimizer is None:
                raise NotImplementedError("optimizer is None")
            instructions = self.optimizer(self.simulator, instructions)
        return instructions

    def __getattr__(self, item):
        # TODO: very strange function...
        # The main difference between __getattr__ and __getattribute__ is that if the attribute was not found by the usual way then __getattr__ is used.
        if item in self.objects:
            return self.objects[item]
        else:
            raise AttributeError
