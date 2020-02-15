import argparse
from robot.envs.hyrule.sim2d import Sim2D, ControlPanelV1

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    sim = Sim2D()
    panel = ControlPanelV1(sim)

    idx = 0
    while True:
        sim.step()
        panel.rot(1).transport('ball')
        panel.step()
        #panel.rot(1).move().step()
        #panel.rot(1).move().step()
        img = sim.render()
        #print(sim.input.get_key_down(ord('k')))

if __name__ ==  '__main__':
    main()

