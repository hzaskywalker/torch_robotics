from .object import Object
from .instruction import ControlPanel

class Relation:
    # constraints are similar to pytorch functions..
    # we maintain
    def __init__(self, timestep, perpetual=True):
        self.timestep = timestep
        self.perpetual = perpetual

    def __call__(self, object: Object, panel: ControlPanel):
        return self.execute(object, panel)

    def execute(self, object: Object, panel: ControlPanel):
        raise NotImplementedError


class Action(Relation):
    def __init__(self, instr, *args, timestep=0, perpetual=True):
        super(Action, self).__init__(timestep, perpetual)
        self.instr = instr
        self.args = args

    def execute(self, object: Object, panel: ControlPanel):
        return panel.execute(self.instr, *self.args, object.pointer)

    def __str__(self):
        return f"Action({str(self.instr)}({' '.join([str(i) for i in self.args])})"

class Constraint(Relation):
    def __init__(self, instr, *args, timestep=1, perpertual=True):
        self.instr = instr
        self.args = args
        super(Constraint, self).__init__(timestep, perpertual)

    def execute(self, object: Object, panel: ControlPanel):
        assert object.parent is not None, "You can't add constrain to the world itself"
        return panel.execute(self.instr, *self.args, object.pointer, object.parent.pointer)

    def __str__(self):
        return f"Constrain({str(self.instr)}({' '.join([str(i) for i in self.args])})"

class Cost:
    def __call__(self, object: Object):
        return self.execute(object)

    def execute(self, object: Object):
        raise NotImplementedError
