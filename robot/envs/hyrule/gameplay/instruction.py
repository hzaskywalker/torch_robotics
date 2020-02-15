"""
The logic looks like this:
    - For magical operator, for example add a link and move the robot, it will return a magic instruction that apply the simulator directly.
        The added magical joints, must be detached by a magical constraints.
    - As for all cost function, it will not be removed unless it's deleted from the tree. We will always evaluate the cost function in every step during the execution.

In short, magical constraints must be canceled by a magical constraints, and the normal constrains will be removed after removing it from the tree.

Ideally, each instruction as a timelabel , we will sort the operators by its starting point... to speed it up
"""
from ..simulator import Simulator
from inspect import isfunction


# low-level instructions that could be exectuted by sapien simulator directly...
class Instruction(object):
    def __call__(self, simulator, step):
        self.forward(simulator, step)

    def forward(self, simulator, step):
        # step == -1 for the end of the horizon
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

    def forward(self, simulator, step):
        if step > 0:
            simulator.agent.set_qf(self.actions[step])

    def __str__(self):
        return "SET_QF actions: T x dof"


# very ugly hack for making function...
class function:
    def __init__(self, func, *args):
        self.func = func
        self.args = args

    def forward(self, simulator, step):
        if step == 0:
            self.func(simulator, *self.args)

    def __str__(self):
        return f"Function call to {self.func} with args {self.args}"


class function_type:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args):
        return function(self.func, *args)


class ControlPanel:
    # Set of instruction that wrap an simulator/controllable into a controllable
    # This defines the interface between the environment and the high-level controller/language
    # horizon is actually the frameskip term..
    def __init__(self, sim, horizon=1):
        self._sim = sim
        self.horizon = horizon
        self.instr_set = {}
        self._instr_cache = []

    @property
    def sim(self):
        if isinstance(self._sim, Simulator):
            return self._sim
        return self._sim.sim

    def step(self, instructions=None):
        if instructions is None:
            instructions = self._instr_cache
        for i in self._instr_cache:
            print(i)
        for i in range(self.horizon):
            for instr in instructions:
                instr(self.sim, i)
            self.sim.do_simulation()
        for instr in instructions:
            instr(self.sim, -1)
        self._instr_cache = []

    def register(self, name, type, *args):
        if isfunction(type):
            type = function_type(type)
        self.instr_set[name] = (type, args)
        return self

    def __getattr__(self, item):
        return_instrunction = item[0] == '_'
        if return_instrunction:
            item = item[1:]
        if item in self.instr_set:
            out = self.instr_set[item][0]
            def run(*args, **kwargs):
                if return_instrunction:
                    return self._instr_cache[-1]
                else:
                    self._instr_cache.append(out(*args, **kwargs))
                    return self
            return run
        elif isinstance(self._sim, ControlPanel):
            return self._sim.__getattr__(item)
        else:
            raise AttributeError(f"No registered instruction {item}")
