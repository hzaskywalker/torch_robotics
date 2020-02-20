import argparse
import time
from robot.envs.hyrule.sim3d import Sim3D

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    sim = Sim3D()
    sim.move_xyz([0.6, 0.3, 0.8])

    while True:
        sim.step()
        print(sim.gripper.pose)
        sim.render()



if __name__ == '__main__':
    main()
