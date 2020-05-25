# we separate the geometry and the low-level physics
# we plan to include several implementation
#   - fcl based collision check
#   - naive pytorch gpu-based implementation

#   - We view the collision checker as a service
#   - One can register the object and mesh into the geometry by add_shape
#   -     we then set pose and ask for the solution


from .simplex import Simplex
