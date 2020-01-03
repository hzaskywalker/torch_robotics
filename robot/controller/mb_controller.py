# Wrapper of the model and model based controller
import gym
import os
import tqdm
from robot.utils.framebuffer import TrajBuffer
import torch
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
                 timestep=100,  #max trajectory length
                 load=False,
                 init_buffer_size=1000, init_train_step=100000,
                 path=None,
                 data_path=None,
                 vis=None,
                 batch_size=200,
                 valid_ratio=0.2,
                 iters_per_epoch=None,
                 valid_batch_num=0,
                 hook=[], data_sampler='random'):
        assert isinstance(model, AgentBase)
        assert isinstance(controller, ForwardControllerBase) or controller is None

        if data_path is None:
            data_path = path
        self.data_path = data_path

        self.data_sampler = data_sampler

        if path is None or not os.path.exists(os.path.join(path, 'agent')) or not load:
            self.loaded_model = False
            self.model = model
        else:
            tmp_model = torch.load(os.path.join(path, 'agent'))
            self.loaded_model = True
            assert type(tmp_model) == type(model)
            self.model = tmp_model

        self.controller = controller
        self.buffer = TrajBuffer(maxlen, batch_size=batch_size, valid_ratio=valid_ratio)

        self.init_flag = False
        self.init_buffer_size = init_buffer_size
        self.init_train_step = init_train_step

        self.timestep = timestep
        self.path = path
        if self.path is not None and not os.path.exists(self.path):
            os.makedirs(self.path)

        self.vis = vis
        self.train_iter = 0
        self.iters_per_epoch = iters_per_epoch
        self.valid_batch_num = valid_batch_num

        self._outputs = []
        self.hook = hook

    # controller part
    def reset(self, cost=None):
        self.controller.set_model(self.model)
        if cost is not None:
            self.controller.set_cost(cost)
        return self.controller.reset()

    def __call__(self, x):
        out = tocpu(self.controller(x))
        return out


    # update part
    def update_network(self, env=None, num_train=50):
        for data in self.buffer.make_sampler(self.data_sampler, 'train', num_train):
            output = self.model.update(*data)
            self._outputs.append(output)
            self.train_iter += 1

            if self.iters_per_epoch is not None and self.train_iter % self.iters_per_epoch == 0:
                self.after_epoch(data, output, env)

    def after_epoch(self, data, output, env=None):
        self.model.eval()

        dic = merge_training_output(self._outputs, 'train')
        self._outputs = [] #clear

        #for fn in train_vis_fn:
        #    fn(train_info, train_outputs, data, output)
        if 'visualize' in self.model.__dir__():
            self.model.visualize('train', data, dic, env)

        if self.valid_batch_num > 0:
            outputs = []
            for data in self.buffer.make_sampler('random', 'valid', self.valid_batch_num):
                outputs.append(self.model.update(*data))
            dic = {**merge_training_output(outputs, 'valid'), **dic}
            if 'visualize' in self.model.__dir__():
                self.model.visualize('valid', data, dic, env)

        for fn in self.hook:
            fn(self, dic)

        if self.vis is not None:
            self.vis(dic)

        self.model.save(self.path)
        self.model.train()
        return dic

    def update_buffer(self, env, policy, num_traj):
        for i in range(num_traj):
            s, a = rollout(env, policy, timestep=self.timestep)
            self.buffer.update(s, a)


    # init and fit
    def init(self, env: gym.Env):
        if self.init_buffer_size > 0:
            print('init....')
            def random_policy(*args, **kwargs):
                return env.action_space.sample()

            # add cache mechanism
            if self.data_path is not None:
                init_buffer_path = os.path.join(self.data_path, 'init_buffer')

            if self.data_path is not None and os.path.exists(init_buffer_path):
                self.buffer.load(init_buffer_path)
            else:
                print('filling buffer...')
                for _ in tqdm.trange(self.init_buffer_size):
                    self.update_buffer(env, random_policy)
                if self.path is not None:
                    self.buffer.save(init_buffer_path)

            print('train network...')
            if not self.loaded_model:
                for _ in tqdm.trange(self.init_train_step):
                    self.update_network(env)
        self.init_flag = True
        return self.buffer, self.model

    def fit(self, env, num_iter=1, num_traj=1, num_train=50):
        if not self.init_flag:
            self.init(env)

        for _ in range(num_iter):
            self.reset()
            self.update_buffer(env, self, num_traj)
            self.update_network(env, num_train)
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

    controller = CEMController(20, env.action_space, model,
                               cost, std=float(env.action_space.high.max() / 3),
                               iter_num=5, num_mutation=80, num_elite=8,
                               mode='fix')

    mb_controller = MBController(model, controller, maxlen=int(1e6), timestep=timestep,
                                 init_buffer_size=200, init_train_step=10000, path='/tmp/xxx/',
                                 vis=Visualizer('/tmp/xxx/history'))

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
