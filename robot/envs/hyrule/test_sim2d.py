import argparse
import time
from robot.envs.hyrule.sim2d import Sim2D, Sim2DV1, Sim2DWorld

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    sim = Sim2DV1()
    world = Sim2DWorld(sim)
    #print(world.ball.pointer)
    #exit(0)
    #world.rot(world.agent, 1).transport(world.ball, parent=world.agent)
    world.rot(world.agent, 1).step().render(1)
    world.move(world.agent).step().render(1)
    world.grasped(world.ball, parent=world.agent).step().render(1)
    world.rot(world.agent, 1).step().render(1)
    world.move(world.agent).step().render(1)
    world.rot(world.agent, -1).step().render(1)
    world.move(world.agent).step().render(1)

    while True:
        world.step()
        world.render()



if __name__ ==  '__main__':
    main()

