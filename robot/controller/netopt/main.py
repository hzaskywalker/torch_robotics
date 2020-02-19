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


def cem(env, num_iter=5, device='cpu'):
    def eval(obs, x):
        x = x.detach().cpu().numpy()
        return torch.tensor(
            env.step(x)['reward'],
            device=device, dtype=torch.float
        )

    optimizer = CEM(eval,
               iter_num=num_iter,
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


def SGD(env, num_iter=100, lr=0.1):
    def fn(scene):
        x0 = np.zeros(shape=(env.n,))
        env.reset(scene)
        momentum = None
        for i in range(num_iter):
            grad = env.step(x0)['grad']

            if momentum is not None:
                momentum = momentum * 0.9 - grad * lr
            else:
                momentum = grad
            x0 = x0 + momentum
        return x0
    return fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", type=str, default='cem', choices=['cem', 'SGD'])
    parser.add_argument("--num_iter", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    env, test_env = make(), make()
    if args.optimizer == 'cem':
        optimizer = cem(env, num_iter=args.num_iter)
    elif args.optimizer == 'SGD':
        optimizer = SGD(env, num_iter=args.num_iter, lr=args.lr)
    else:
        raise NotImplementedError

    print('result', test(test_env, optimizer))



if __name__ == '__main__':
    main()