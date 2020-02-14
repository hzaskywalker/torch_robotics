"""
The logic looks like this:
    - For magical operator, for example add a link and move the robot, it will return a magic instruction that apply the simulator directly.
        The added magical joints, must be detached by a magical constraints.
    - As for all cost function, it will not be removed unless it's deleted from the tree. We will always evaluate the cost function in every step during the execution.

In short, magical constraints must be canceled by a magical constraints, and the normal constrains will be removed after removing it from the tree.

Ideally, each instruction as a timelabel , we will sort the operators by its starting point... to speed it up
"""


# low-level instructions that could be exectuted by sapien simulator directly...
class Instruction(object):
    def __init__(self, start=0, end=None):
        self.start = start
        self.end = end

    def __call__(self, simulator, step):
        self.forward(simulator, step)

    def forward(self, simulator, step):
        raise NotImplementedError

# hard setting something...
class Magic(Instruction):
    pass

# costs that could be evaluated by sapien simulator directly...
class Cost(Instruction):
    pass
