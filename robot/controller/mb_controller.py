# Wrapper of the model and model based controller
import gym
import os
import tqdm
from robot.model.framebuffer import TrajBuffer
from robot.utils import rollout, AgentBase, tocpu, evaluate
from robot.controller.forward_controller import ForwardControllerBase


class MBController:
    """
    not end2end model-based controller
    contains the following part:
        model_agent:
            1. network and its optimizer
            2. a framebuffer where we can use it to udpate the model
        controller:
            1. take model as a parameter and output the actions
    """
    def __init__(self, model, controller, maxlen,
                 timestep=100, #max trajectory length
                 init_buffer_size=1000, init_train_step=100000,
                 cache_path=None, vis=None):
        assert isinstance(model, AgentBase)
        assert isinstance(controller, ForwardControllerBase)

        self.model = model
        self.controller = controller
        self.buffer = TrajBuffer(maxlen)

        self.init_flag = False
        self.init_buffer_size = init_buffer_size
        self.init_train_step = init_train_step

        self.timestep = timestep
        self.cache_path = cache_path
        if self.cache_path is not None and not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        self.vis = vis

    def update_network(self, iters=1):
        output = None
        for i in range(iters):
            data = self.buffer.sample('train')
            output = self.model.update(*data)
            if self.vis is not None:
                self.vis(output)
        return output

    def update_buffer(self, env, policy):
        s, a = rollout(env, policy, timestep=self.timestep)
        self.buffer.update(s, a)

    # controller part
    def reset(self, cost=None):
        self.controller.set_model(self.model)
        if cost is not None:
            self.controller.set_cost(cost)
        return self.controller.reset()

    def __call__(self, x):
        out = tocpu(self.controller(x))
        return out

    def init(self, env: gym.Env):
        if self.init_buffer_size > 0:
            print('init....')
            def random_policy(*args, **kwargs):
                return env.action_space.sample()

            # add cache mechanism
            init_buffer_path = os.path.join(self.cache_path, 'init_buffer')
            if self.cache_path is not None and os.path.exists(init_buffer_path):
                self.buffer.load(init_buffer_path)
            else:
                print('filling buffer...')
                for _ in tqdm.trange(self.init_buffer_size):
                    self.update_buffer(env, random_policy)
                if self.cache_path is not None:
                    self.buffer.save(init_buffer_path)

            print('train network...')
            for _ in tqdm.trange(self.init_train_step):
                self.update_network()
        self.init_flag = True
        return self.buffer, self.model

    def fit(self, env, num_iter=1, num_train=50):
        if not self.init_flag:
            self.init(env)

        for _ in range(num_iter):
            self.reset()
            self.update_buffer(env, self)
            for _ in range(num_train):
                self.update_network()
        return self

    def test(self, env, num_episode=10):
        return evaluate(env, self, self.timestep, num_episode)


def test_mb_controller():
    import torch
    import numpy as np
    from robot.envs import make
    from robot.model.mlp_forward import MLPForward
    from robot.controller.forward_controller import GDController, CEMController
    from robot.utils import Visualizer
    env = make("CartPole-v0")
    timestep = 100

    model = MLPForward(0.001, env, num_layer=3, feature=256, batch_norm=False).cuda()

    def cost(s, a, t, it):
        x, dx, th, dth = torch.unbind(t, dim=1)
        th_target = 20 / 360 * np.pi
        x_target = 2.2
        out = ((th - 0) ** 2) + ((th_target - 0) ** 2)  # this loss is much easier to optimize
        return out
        """
        return (torch.nn.functional.relu(th - th_target) ** 2) + \
               (torch.nn.functional.relu(-th_target - (-th)) ** 2) + \
               (torch.nn.functional.relu(x - x_target) ** 2) + \
               (torch.nn.functional.relu(-x_target - (-x)) ** 2)
               """

    #controller = GDController(timestep, env.action_space, model, cost, lr=0.001)

    controller = CEMController(20, env.action_space, model,
                               cost, std=float(env.action_space.high.max() / 3),
                               iter_num=5, num_mutation=80, num_elite=8,
                               mode='fix')
    mb_controller = MBController(model, controller, maxlen=int(1e6), timestep=timestep,
                                 init_buffer_size=200, init_train_step=10000,  cache_path='/tmp/xxx/', vis=Visualizer('/tmp/xxx/history'))

    mb_controller.init(env)
    print('testing...')
    print(mb_controller.test(env))

    for i in range(1000):
        mb_controller.fit(env)
        if i % 10 == 0:
            print('testing...')
            print(mb_controller.test(env))



if __name__ == '__main__':
    test_mb_controller()
