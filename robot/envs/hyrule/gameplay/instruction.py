"""
The logic looks like this:
    - For magical operator, for example add a link and move the robot, it will return a magic instruction that apply the simulator directly.
        The added magical joints, must be detached by a magical constraints.
    - As for all cost function, it will not be removed unless it's deleted from the tree. We will always evaluate the cost function in every step during the execution.

In short, magical constraints must be canceled by a magical constraints, and the normal constrains will be removed after removing it from the tree.

Ideally, each instruction as a timelabel , we will sort the operators by its starting point... to speed it up
"""
from ..simulator import Simulator


# low-level instructions that could be exectuted by sapien simulator directly...
class Instruction(object):
    def __call__(self, simulator, step):
        self.forward(simulator, step)

    def forward(self, simulator, step):
        # step == -1 for the end of the horizon
        raise NotImplementedError

    def __str__(self):
        return "Instruction base class, please implement it...>>>>>>>>>>>>>>>>>>>"

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

    def forward(self, simulator, step):
        if step > 0:
            simulator.agent.set_qf(self.actions[step])

    def __str__(self):
        return "set_qf actions: T x dof"


class call_func:
    def __init__(self, func):
        self.func = func
    def forward(self, simulator, step):
        self.func(simulator, step)

class ControlPanel:
    # Set of instruction that wrap an simulator/controllable into a controllable
    # This defines the interface between the environment and the high-level controller/language
    # horizon is actually the frameskip term..
    def __init__(self, sim, horizon=1):
        self._sim = sim
        self.horizon = horizon
        self.instructions = {}

    @property
    def sim(self):
        if isinstance(self._sim, Simulator):
            return self._sim
        return self._sim.sim

    def step(self, instructions=None):
        if instructions is None:
            instructions = []
        for i in range(self.horizon):
            for instr in instructions:
                instr(self.sim, i)
            self.sim.do_simulation()
        for instr in instructions:
            instr(self.sim, -1)

    def register(self, name, instrunction):
        if isinstance(instrunction, Instruction):
            self.instructions[name] = instrunction
        else:
            self.instructions[name] = call_func(instrunction)
        return self
