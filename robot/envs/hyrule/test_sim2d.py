import argparse
from robot.envs.hyrule.sim2d import Sim2D, ControlPanelV1

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    sim = Sim2D()
    panel = ControlPanelV1(sim)

    while True:
        panel.rot(1).step().transport(sim.ball)
        #panel.step().transport(sim.ball)
        panel.render()

if __name__ ==  '__main__':
    main()

