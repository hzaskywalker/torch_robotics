import argparse
from robot.envs.hyrule import Simulator, load_scene, load_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    simulator = Simulator()
    params = load_json(args.path)
    load_scene(simulator, params)

    if 'trajectories' in params:
        trajectories = params['trajectories']
        while True:
            for i in trajectories:
                for state in i[0]:
                    simulator.load_state_vector(state)
                    simulator.render()

    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
