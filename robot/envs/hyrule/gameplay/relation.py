from .object import Object
from .instruction import ControlPanel

class Relation:
    # constraints are similar to pytorch functions..
    # we maintain
    def __call__(self, object: Object):
        return self.parse(object)

    def parse(self, object: Object):
        raise NotImplementedError


class Action(Relation):
    def __init__(self, instr, *args):
        self.instr = instr
        self.args = args

    def parse(self, object: Object):
        return self.instr(*self.args, object.pointer)

class Constrain(Relation):
    def __init__(self, instr, *args):
        self.instr = instr
        self.args = args

    def parse(self, object: Object):
        assert object.parent is not None, "You can't add constrain to the world itself"
        return self.instr(*self.args, object.pointer, object.parent.pointer)
