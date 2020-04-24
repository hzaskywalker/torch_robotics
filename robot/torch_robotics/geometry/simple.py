# this the most simple collision checker, we support the collision between the following objects
#   - mesh: triangles or squares that with normal
#   - sphere:

class RigidBody:
    def __init__(self):
        pass

class Sphere:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius


class Ground:
    def __init__(self):
        pass


class Service:
    def __init__(self):
        pass

    def register(self, shape):
        pass

    def sphere(self, center):
        pass
