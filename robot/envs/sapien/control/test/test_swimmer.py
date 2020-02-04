import sys
import tqdm
from robot.envs.sapien import SwimmerEnv

def test():
    x = 0 if len(sys.argv) > 1 else 1
    swimmer = SwimmerEnv()
    print(swimmer.observation_space)
    print(swimmer.action_space)

    swimmer.reset()
    for i in tqdm.trange(10000):
        action = swimmer.action_space.sample()
        swimmer.step(action * x)
        #swimmer.render()


if __name__ == '__main__':
    test()
