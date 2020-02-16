class OptimizerBase:
    def __init__(self, world, parameters):
        self.parameters = parameters
        self.world = world

    def step(self):
        raise NotImplementedError
