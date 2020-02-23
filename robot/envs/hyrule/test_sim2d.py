import argparse
import time
from robot.envs.hyrule.sim2d import Sim2D, Fixed

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    world = Sim2D()
    #print(world.ball.pointer)
    #exit(0)
    world.rot(world.agent, 1).step().render(sleep=1)
    world.move(world.agent).step().render(sleep=1)
    world.fixed(world.ball, world.agent).step().render(sleep=1)
    world.rot(world.agent, 1).step().render(sleep=1)
    world.move(world.agent).step().render(sleep=1)
    world.rot(world.agent, -1).step().render(sleep=1)
    world.move(world.agent).step().render(sleep=1)
    world.rot(world.agent, -1).step().render(sleep=1)
    world.move(world.agent).step().render(sleep=1)
    world.move(world.agent).step().render(sleep=1)
    for i in world._constraints:
        if isinstance(i, Fixed):
            world.remove_constraints(i)
    world.rot(world.agent, -1).step().render(sleep=1)
    world.move(world.agent).step().render(sleep=1)
    while True:
        world.step()
        world.render()

if __name__ ==  '__main__':
    main()

