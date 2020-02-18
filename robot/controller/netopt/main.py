import torch
import numpy as np
import argparse
from robot.controller.netopt.leastsquare import LeastSqaure
from robot.controller.cem import CEM

N_TEST = 20


def make():
    return LeastSqaure(5, 3)


def test(env, func, num_test=200):
    outputs = []
    for i in range(num_test):
        scene = env.reset()
        out = func(scene)
        outputs.append(env.step(out)['reward'])
    return np.array(outputs).mean()


def cem(env, device='cpu'):
    def eval(obs, x):
        x = x.detach().cpu().numpy()
        return torch.tensor(
            env.step(x)['reward'],
            device=device, dtype=torch.float
        )

    optimizer = CEM(eval,
               iter_num=5,
               num_mutation=100,
               num_elite=10,
               std=0.2,
               alpha=0.)

    def fn(scene):
        x0 = torch.zeros(size=(env.n,), dtype=torch.float, device=device)
        env.reset(scene)
        output = optimizer(scene, x0)
        return output.detach().cpu().numpy()

    return fn


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env, test_env = make(), make()
    optimizer = cem(env)
    print('result', test(test_env, optimizer))



if __name__ == '__main__':
    main()