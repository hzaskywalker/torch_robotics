"""
The logic looks like this:
    - For magical operator, for example add a link and move the robot, it will return a magic instruction that apply the simulator directly.
        The added magical joints, must be detached by a magical constraints.
    - As for all cost function, it will not be removed unless it's deleted from the tree. We will always evaluate the cost function in every step during the execution.

In short, magical constraints must be canceled by a magical constraints, and the normal constrains will be removed after removing it from the tree.

Ideally, each instruction as a timelabel , we will sort the operators by its starting point... to speed it up

The format of the instruction name is:
    *args, object_name, [object_parent_name]
"""
# low-level instructions that could be exectuted by sapien simulator directly...
class Instruction(object):
    def __call__(self, simulator):
        self.forward(simulator)

    def forward(self, simulator):
        raise NotImplementedError

    def __str__(self):
        return f"Instruction base class, please implement it...>>>>>>>>>>>>>>>>>>>{self.__class__}"

# hard setting something...
class Magic(Instruction):
    pass

# costs that could be evaluated by sapien simulator directly...
class Cost(Instruction):
    pass


class _set_qf(Magic):
    def __init__(self, actions):
        super(_set_qf, self).__init__()
        self.actions = actions

    def forward(self, simulator):
        # apply before
        simulator.agent.set_qf(self.actions)

    def __str__(self):
        return "SET_QF actions: T x dof"


# very ugly hack for making function...
class function:
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def forward(self, simulator):
        self.func(simulator, *self.args)

    def __str__(self):
        return f"Function call to {self.func} with args {self.args}"


class function_type:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args):
        return function(self.func, *args)
