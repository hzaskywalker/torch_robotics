import argparse
import numpy as np
from robot.envs.hyrule import load_json, dump_json


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    base = load_json('x.json')
    for i in range(500):
        if i>0:
            base['box']['size'] = list(np.random.random(size=(3,)) * 0.04 +np.array([0.01, 0.01, 0.01]))
            x = np.random.random() * (1.1 - 0.5) + 0.5
            y = np.random.random() * (0.3 + 0.3) - 0.3
            z = base['box']['size'][-1] * 2  + 0.5
            base['box']['center'] = [x, y, z]

            x = np.random.random() * (1.0 - 0.5) + 0.5
            y = np.random.random() * (0.3 + 0.3) - 0.3
            z = np.random.random() * 0.5  + 0.5
            base['waypoints'][0]['list'][1][1]['target'] = [x, y, z]

            x = np.random.random() * (1.0 - 0.5) + 0.5
            y = np.random.random() * (0.3 + 0.3) - 0.3
            z = np.random.random() * 0.5  + 0.5
            base['waypoints'][1]['list'][1][1]['target'] = [x, y, z]

        dump_json(f'scenes/{i}.json', base)

if __name__ == '__main__':
    main()