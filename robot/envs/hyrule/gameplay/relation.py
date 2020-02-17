from .object import Object
from .simulator import Simulator

class Relation:
    # constraints are similar to pytorch functions..
    # we maintain
    def __init__(self, timestep, perpetual=True):
        self.timestep = timestep
        self.perpetual = perpetual

    def __call__(self, object: Object, sim: Simulator):
        return self.execute(object, sim)

    def execute(self, object: Object, sim: Simulator):
        """
        :param object:
        :param sim:
        :return: 0 if the instruction success otherwise the relation will be terminated..
        """
        raise NotImplementedError

    def prerequisites(self, object: Object, parent: Object):
        # determine if the we can add one relation at that moment.
        raise NotImplementedError


class Action(Relation):
    def __init__(self, instr, *args, timestep=0, perpetual=True):
        super(Action, self).__init__(timestep, perpetual)
        self.instr = instr
        self.args = args

    def execute(self, object: Object, sim: Simulator):
        return sim.execute(self.instr, *self.args, object.pointer)

    def __str__(self):
        return f"Action({str(self.instr)}({' '.join([str(i) for i in self.args])})"

    def prerequisites(self, object: Object, parent: Object):
        return True

class Constraint(Relation):
    def __init__(self, instr, *args, timestep=1, perpertual=True):
        self.instr = instr
        self.args = args
        super(Constraint, self).__init__(timestep, perpertual)

    def execute(self, object: Object, sim: Simulator):
        assert object.parent is not None, "You can't add constrain to the world itself"
        return sim.execute(self.instr, *self.args, object.pointer, object.parent.pointer)

    def prerequisites(self, object: Object, parent: Object):
        return True

    def __str__(self):
        return f"Constrain({str(self.instr)}({' '.join([str(i) for i in self.args])})"

class Cost:
    def __call__(self, object: Object):
        return self.execute(object)

    def execute(self, object: Object):
        raise NotImplementedError
