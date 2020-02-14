import argparse
from robot.envs.hyrule.sim2d import Sim2D

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    sim = Sim2D()

    while True:
        sim.step()
        img = sim.render()

if __name__ ==  '__main__':
    main()

