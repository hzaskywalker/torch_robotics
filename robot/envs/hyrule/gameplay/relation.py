from .object import Object
from .instruction import ControlPanel

class Relation:
    # constraints are similar to pytorch functions..
    # we maintain
    def __init__(self, timestep):
        self.timestep = timestep

    def __call__(self, object: Object, panel: ControlPanel):
        return self.execute(object, panel)

    def execute(self, object: Object, panel: ControlPanel):
        raise NotImplementedError


class Action(Relation):
    def __init__(self, instr, *args, timestep=0):
        self.instr = instr
        self.args = args
        super(Action, self).__init__(timestep)

    def execute(self, object: Object, panel: ControlPanel):
        return self.instr(*self.args, object.pointer)

class Constraint(Relation):
    def __init__(self, instr, *args, timestep=1):
        self.instr = instr
        self.args = args
        super(Constraint, self).__init__(timestep)r

    def execute(self, object: Object, panel: ControlPanel):
        assert object.parent is not None, "You can't add constrain to the world itself"
        return self.instr(*self.args, object.pointer, object.parent.pointer)


class Cost:
    def __call__(self, object: Object):
        return self.execute(object)

    def execute(self, object: Object):
        raise NotImplementedError
