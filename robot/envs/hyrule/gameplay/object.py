from typing import List, Set
from .relation import Relation

class Object(object):
    # object stores all the information of the current object
    # each object is associated with the
    # we don't handle the name of the object by the object itself, as name is something that other people call it.
    def __init__(self, pointer, parent=None):
        self.pointer = pointer
        self.parent = parent

        if self.parent is not None:
            self.world = self.parent.world # store the root of the tree
        else:
            self.world = self
            self._simulator = pointer

        self.child: Set[Object] = set()
        # maintain the tree by the previous values

        self.relation: List[Relation]  = [] # only store the constraints to parent, or null if it's a free element in the world

    @property
    def simulator(self):
        if self.world == self:
            return self._simulator
        return self.world._simulator

    def add_child(self, element):
        self.child.add(element)

    def remove_child(self, element):
        self.child.remove(element)

    def linkto(self, parent):
        if parent == self.parent:
            return
        if self.parent is not None:
            self.parent.remove_child(self)

        self.parent = parent
        self.parent.add_child(self)

    def add_relation(self, parent, relation):
        if self.parent is not None and self.parent != parent:
            raise Exception("Must detach before adding new constraint")
        self.linkto(parent)
        self.relation.append(relation)

    def remove_relation(self, relation):
        # pass discard some constraints...
        self.relation.remove(relation)
        if len(self.relation) == 0:
            self.linkto(self.world)

    def parse(self):
        instructions = []
        for rel in self.relation:
            instructions.append(rel.parse(self))

        instructions = []
        for obj in self.child:
            instructions.append(obj.parse())
        return instructions
