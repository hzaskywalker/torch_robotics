from typing import List, Set
import logging

class Object(object):
    # object stores all the information of the current object
    # each object is associated with the
    # we don't handle the name of the object by the object itself, as name is something that other people call it.

    def __init__(self, pointer, parent=None):
        self.pointer = pointer
        self.parent: Object = parent

        if self.parent is not None:
            self.world = self.parent.world # store the root of the tree
        else:
            self.world = self

        self.child: Set[Object] = set()
        # maintain the tree by the previous values

        from .relation import Relation, Cost
        self.relations: List[Relation] = [] # only store the constraints to parent, or null if it's a free element in the world
        self.costs: List[Cost] = []

        self._reward = None

    @property
    def name(self):
        if self.pointer is not None:
            return self.pointer.name
        else:
            return 'world'

    def add_child(self, element):
        self.child.add(element)

    def remove_child(self, element):
        self.child.remove(element)

    def linkto(self, parent):
        if parent == self.parent:
            return
        print('link...', self.name, parent.name)
        if self.parent is not None:
            self.parent.remove_child(self)
        self.world = parent.world
        self.parent = parent
        self.parent.add_child(self)

    def add_relation(self, relation, parent=None):
        if not relation.prerequisites(self, parent):
            logging.warning("adding relation failed")
            return
        if parent is not None:
            if self.parent is not None and self.parent is not parent and len(self.relations) > 0:
                raise Exception("Must detach before adding new constraint")
            self.linkto(parent)
        self.relations.append(relation)
        return

    def remove_relation(self, relation):
        # pass discard some constraints...
        self.relations.remove(relation)
        if len(self.relations) == 0:
            self.linkto(self.world)

    def execute(self, sim, timestep):
        to_remove = []
        for rel in self.relations:
            if rel.timestep == timestep:
                tmp = rel(self, sim)
                if tmp or not rel.perpetual:
                    print('xxx', rel, tmp, rel.perpetual)
                    to_remove.append(rel)
        for i in to_remove:
            self.remove_relation(i)

    def forward(self, sim):
        #print(self.name, [i.name for i in self.child])
        self.execute(sim, 0)
        for obj in list(self.child):
            obj.forward(sim)

    def backward(self, sim):
        # backward is the function called after step function
        self.execute(sim, 1)
        reward = 0
        for obj in list(self.child):
            reward += obj.backward(sim)
        for cost in self.costs:
            reward += cost(self)
        self._reward = reward # store the _reward in this object for mpc
        return reward
