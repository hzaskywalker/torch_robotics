from .physical_object import PhysicalObject

class Imortal(PhysicalObject):
    xdof = 0
    vdof = 0

    def fk(self):
        return None
