# Torch version of model
# high level control instruction... to revise the world object...But I don't think they will be very useful now, as we can directly change the world object directly.
# It will transfer the

from .world import World

class Program:
    def __call__(self, world):
        assert isinstance(world, World), "the input to a piece of program must be a World object"
        self.forward(world)

    def forward(self, world):
        raise NotImplementedError

# control command .. not very useful now as if we don't need to optimize the conditions, we can directly use python
# instead of IF, WHILE here.. I write those here as an example..

class Sequential(Program):
    def __init__(self, *args):
        self.args = args

    def forward(self, world):
        for i in self.args:
            i.forward(world)


class IF(Program):
    def __init__(self, condition, body, elseif=None):
        Program.__init__(self)
        self.condition = condition
        self.body = body
        self.elseif = elseif

    def forward(self, world):
        if self.condition(world):
            self.body(world)
        if self.elseif is not None:
            self.elseif(world)

class WHILE(Program):
    def __init__(self, condition, body):
        super(WHILE, self).__init__()
        self.condition = condition
        self.body = body

    def forward(self, world):
        while self.condition(world):
            self.body(world)
