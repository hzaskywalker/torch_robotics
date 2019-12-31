# Wrapper of the model and model based controller
import gym
import os
import tqdm
from robot.utils.framebuffer import TrajBuffer
from robot.utils import rollout, AgentBase, tocpu, evaluate
from robot.controller.forward_controller import ForwardControllerBase
from robot.utils.trainer import merge_training_output


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
                 cache_path=None, vis=None,
                 batch_size=200,
                 valid_ratio=0.2, episode=None, valid_batch_num=0):
        assert isinstance(model, AgentBase)
        assert isinstance(controller, ForwardControllerBase) or controller is None

        self.model = model
        self.controller = controller
        self.buffer = TrajBuffer(maxlen, batch_size=batch_size, valid_ratio=valid_ratio)

        self.init_flag = False
        self.init_buffer_size = init_buffer_size
        self.init_train_step = init_train_step

        self.timestep = timestep
        self.cache_path = cache_path
        if self.cache_path is not None and not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)

        self.vis = vis
        self.train_iter = 0
        self.episode = episode
        self.valid_batch_num = valid_batch_num

        self._outputs = []

    def update_network(self):
        data = self.buffer.sample('train')
        output = self.model.update(*data)
        self._outputs.append(output)
        self.train_iter += 1

        if self.episode is not None and self.train_iter % self.episode == 0:
            self.after_epoch()
        return output

    def after_epoch(self):
        self.model.eval()

        dic = merge_training_output(self._outputs, 'train')
        self._outputs = [] #clear

        if self.valid_batch_num > 0:
            outputs = []
            for i in range(self.valid_batch_num):
                outputs.append(self.model.update(*self.buffer.sample('valid')))
            dic = {**merge_training_output(outputs, 'valid'), **dic}

        if self.vis is not None:
            self.vis(dic)

        # TODO: visualization...

        self.model.train()
        return dic

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
            if self.cache_path is not None:
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
    from robot.controller.forward_controller import CEMController
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
