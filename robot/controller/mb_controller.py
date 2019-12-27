# Wrapper of the model and model based controller
import gym
import tqdm
from robot.model.framebuffer import TrajBuffer
from robot.utils import rollout

class MBController:
    """
    not end2end model-based controller
    contains the following part:
        model_agent:
            1. network and its optimizer
            2. a framebuffer where we can use it to udpate the model
        controller:
            1. take model as a parameter and output the actions

        main_loop function that will handle the training and model saving.
    """
    def __init__(self, model, controller, maxlen,
                 timestep=100, #max trajectory length
                 init_buffer_size=1000, init_train_step=100000,
                 cache_path=None):
        self.model = model
        self.controller = controller
        self.buffer = TrajBuffer(maxlen)

        self.init_flag = False
        self.init_buffer_size = init_buffer_size
        self.init_train_step = init_train_step

        self.timestep = timestep
        self.cache_path = cache_path

    def update_network(self, iters=1):
        output = None
        for i in range(iters):
            data = self.buffer.sample('train')
            output = self.model.udpate(*data)
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
        return self.controller(x)

    def init(self, env: gym.Env):
        if self.init_buffer_size > 0:
            print('init....')
            def random_policy(*args, **kwargs):
                return env.action_space.sample()

            print('filling buffer...')
            for _ in range(self.init_buffer_size):
                self.update_buffer(env, random_policy)

            print('train network...')
            for _ in range(self.init_train_step):
                self.update_network()
        return self.buffer, self.model

    def fit(self, env, num_iter=1, num_train=50):
        if not self.init_flag:
            self.init(env)

        for _ in tqdm.trange(num_iter):
            self.reset()
            self.update_buffer(env, self)
            for _ in range(num_train):
                self.update_network()
        return self

def test_mb_controller():
    import torch
    import numpy as np
    from robot.envs import make
    from robot.model.mlp_forward import MLPForward
    from robot.controller.forward_controller import GDController
    env = make("CartPole-v0")
    timestep = 200

    model = MLPForward(0.001, env, num_layer=3, feature=256, batch_norm=False)

    def cost(s, a, t, it):
        x, dx, th, dth = torch.unbind(t, dim=1)
        th_target = 20 / 360 * np.pi
        x_target = 2.2
        return (torch.nn.functional.relu(th - th_target) ** 2).sum() + \
               (torch.nn.functional.relu(-th_target - (-th)) ** 2).sum() + \
               (torch.nn.functional.relu(x - x_target) ** 2).sum() + \
               (torch.nn.functional.relu(-x_target - (-x)) ** 2).sum()

    controller = GDController(timestep, env.action_space, model, cost, lr=0.001)
    mb_controller = MBController(model, controller, maxlen=int(1e6), timestep=timestep, init_buffer_size=20, init_train_step=100)
    mb_controller.fit(env)



if __name__ == '__main__':
    test_mb_controller()
