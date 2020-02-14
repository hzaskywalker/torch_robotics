# Torch version of model
# high level control instruction... to revise the world object...But I don't think they will be very useful now, as we can directly change the world object directly.

class Program:
    def __call__(self, parser):
        self.forward(parser)

    def forward(self, parser):
        raise NotImplementedError

# control command .. not very useful now as if we don't need to optimize the conditions, we can directly use python
# instead of IF, WHILE here.. I write those here as an example..

class IF(Program):
    def __init__(self, condition, body, elseif=None):
        Program.__init__(self)
        self.condition = condition
        self.body = body
        self.elseif = elseif

    def __call__(self, world):
        if self.condition(world):
            self.body(world)
        if self.elseif is not None:
            self.elseif(world)

class WHILE(Program):
    def __init__(self, condition, body):
        super(WHILE, self).__init__()
        self.condition = condition
        self.body = body

    def __call__(self, world):
        while self.condition(world):
            self.body(world)
