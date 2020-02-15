import argparse
from robot.envs.hyrule.sim2d import Sim2D, ControlPanelV1

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    sim = Sim2D()
    panel = ControlPanelV1(sim)
    print(panel)
    exit(0)

    while True:
        sim.step()
        img = sim.render()

if __name__ ==  '__main__':
    main()

