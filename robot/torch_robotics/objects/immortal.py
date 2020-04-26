from .physics import Physics

class Imortal(Physics):
    xdof = 0
    vdof = 0

    def fk(self):
        return None
