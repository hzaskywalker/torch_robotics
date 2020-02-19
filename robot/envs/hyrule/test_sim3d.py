import argparse
import time
from robot.envs.hyrule.sim3d import Sim3D

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    sim = Sim3D()

    while True:
        sim.step()
        sim.render()



if __name__ == '__main__':
    main()
