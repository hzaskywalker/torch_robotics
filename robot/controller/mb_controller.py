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
        #assert isinstance(controller, ForwardControllerBase) or controller is None

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
    def update_network(self, env=None, num_train=50, progress=False):
        if 'fit_normalizer' in self.model.__dir__():
            #TODO: very very ugly
            if self.model.normalizer:
                print('fitting...')
                data_gen = self.buffer.make_sampler(self.data_sampler, 'train', 1, use_tqdm=False)
                self.model.fit_normalizer(data_gen)
                print('training...')

        for data in self.buffer.make_sampler(self.data_sampler, 'train', num_train, use_tqdm=progress):
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

    def update_buffer(self, env, policy, num_traj, progress=False, progress_rollout=False):
        ran = tqdm.trange if progress else range
        total = 0
        for _ in ran(num_traj):
            s, a, r = rollout(env, policy, timestep=self.timestep, use_tqdm=progress_rollout)
            self.buffer.update(s, a)
            total += r.sum()
        return total/num_traj # return average reward during update buffer


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
                self.update_buffer(env, random_policy, self.init_buffer_size)
                if self.path is not None:
                    self.buffer.save(init_buffer_path)


            print('train network...')
            if not self.loaded_model:
                self.update_network(env, self.init_train_step, progress=True)

        self.init_flag = True
        return self.buffer, self.model

    def fit(self, env, num_traj=1, num_train=50,
                progress_buffer_update=False, progress_rollout=False, progress_train=False):
        if not self.init_flag:
            self.init(env)

        self.reset()
        if progress_buffer_update or progress_rollout:
            print('update buffer...')
        avg_reward = self.update_buffer(env, self, num_traj, progress_buffer_update, progress_rollout)
        if progress_train:
            print('training...')
        self.update_network(env, num_train, progress=progress_train)
        return {
            'avg_reward': avg_reward
        }

    def test(self, env, num_episode=10, use_tqdm=False, print_reward=False):
        return evaluate(env, self, self.timestep, num_episode, use_tqdm=use_tqdm, print_reward=print_reward)

