import argparse
from robot.envs.hyrule.sim2d import Sim2D, ControlPanelV1, Sim2DWorld

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    sim = Sim2D()
    panel = ControlPanelV1(sim)
    world = Sim2DWorld(sim, panel)
    #print(world.ball.pointer)
    #exit(0)
    world.rot(world.agent, 1).transport(world.ball, parent=world.agent)

    while True:
        #panel.rot(1).step().transport(sim.ball)
        #panel.step().transport(sim.ball)
        world.step()
        panel.render()

if __name__ ==  '__main__':
    main()

