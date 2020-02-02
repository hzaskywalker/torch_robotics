import torch
import copy
import numpy as np
#from tools import Field
#from tools.rl.actor import make_actor
#from tools.utils import soft_update, sync_networks, sync_grads
#from mpi4py import MPI
from .models import actor, critic
from robot.utils.normalizer import Normalizer
from robot.utils import soft_update
from .replay_buffer import ReplayBuffer
from .her import HERSampler
import multiprocessing

# get the flat grads or params
def _get_flat_params_or_grads(network, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])


def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params
    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters():
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()


class AsyncDDPGAgent:
    def __init__(self, observation_space, action_space,
                 gamma=0.99, lr=3e-4, critic_lr=None, tau=0.05, update_target_period=1,
                 normalizer=None, clip_critic=0, device='cpu', pipe=None):

        self.device = device
        inp_dim = observation_space['observation'].shape[0] + observation_space['goal'].shape[0]
        action_dim, self.action_max = action_space.shape[0], action_space.high.max()

        self.critic = critic(inp_dim + action_dim, self.action_max)
        self.actor = actor(inp_dim, action_dim, self.action_max)
        self.target_critic, self.target_actor = None, None
        if normalizer:
            self.normalizer = Normalizer((inp_dim,))
        else:
            self.normalizer = None
        self.clip_critic = clip_critic

        critic_lr = critic_lr if critic_lr is not None else lr

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr)  # 之后之间换个decorator就好了
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), critic_lr)  # 之后之间换个decorator就好了
        self.gamma = gamma
        self.tau, self.update_target_period = tau, update_target_period

        self.update_step = 0

        self.pipe = pipe

    def set_params(self, params):
        assert isinstance(params, dict)
        _set_flat_params_or_grads(self.actor, params['actor'], mode='params'),
        _set_flat_params_or_grads(self.critic, params['critic'], mode='params')

    def get_params(self):
        return {
            'actor': _get_flat_params_or_grads(self.actor, mode='params'),
            'critic': _get_flat_params_or_grads(self.critic, mode='params')
        }

    def sync_grads(self, net):
        if self.pipe is not None:
            grad = _get_flat_params_or_grads(net, mode='grad')
            self.pipe.send(grad)
            grad = self.pipe.recv()
            _set_flat_params_or_grads(net, grad, mode='grad')


    def update_normalizer(self, obs):
        if self.pipe is not None:
            data= [obs.sum(axis=0), (obs**2).sum(axis=0), obs.shape[0]]
            self.pipe.send(data)
            data = self.pipe.recv()
            self.normalizer.update(*data)


    def _update(self, obs, action, t, reward, done):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
            action = torch.tensor(action, dtype=torch.float, device=self.device)
            t = torch.tensor(t, dtype=torch.float, device=self.device)
            reward = torch.tensor(reward, dtype=torch.float, device=self.device)
            if isinstance(done, np.ndarray):
                done = torch.tensor(done, dtype=torch.float, device=self.device)

        # update actor
        if self.target_actor is None:
            self.target_critic = copy.deepcopy(self.critic)
            self.target_actor = copy.deepcopy(self.actor)

        if self.normalizer is not None:
            obs = self.normalizer(obs)
            t = self.normalizer(t)

        with torch.no_grad():
            nextV = self.target_critic(t, self.target_actor(t))
            target = nextV * (1-done) * self.gamma + reward
            if self.clip_critic:
                target = target.clamp(-1/(1-self.gamma), 0) # clip the q value

        predict = self.critic(obs, action)
        assert predict.shape == target.shape
        critic_loss = torch.nn.functional.mse_loss(predict, target)

        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.sync_grads(self.critic)
        self.optim_critic.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.sync_grads(self.actor)
        self.optim_actor.step()

        if (self.update_step+1) % self.update_target_period == 0:
            soft_update(self.target_actor, self.actor, self.tau)
            soft_update(self.target_critic, self.target_critic, self.tau)

        out = dict(
            critic_loss = critic_loss.detach().cpu().numpy(),
            actor_loss = actor_loss.detach().cpu().numpy(),
        )
        return out

    def __call__(self, inp):
        with torch.no_grad():
            if isinstance(inp, np.ndarray):
                inp = torch.tensor(inp, dtype=torch.float, device=self.device)
            inp = inp[None,:]
            if self.normalizer is not None:
                inp = self.normalizer(inp)
            return self.actor(inp)[0]


class Worker(multiprocessing.Process):
    START = 1
    EXIT = 2
    ASK = 3
    GET_PARAM = 4
    SET_PARAM = 5

    def __init__(self, make, env_name, T, replay_buffer_size=int(1e6),
                 noise_eps=0.1, random_eps=0.2,
                 batch_size=256, n_batchs=50, future_K=4, **kwargs):
        super(Worker, self).__init__()
        self.make = make
        self.env_name = env_name
        self.T = T
        self.replay_buffer_size = replay_buffer_size
        self.noise_eps = noise_eps
        self.random_eps = random_eps
        self.batch_size = batch_size
        self.n_batchs = n_batchs

        self.future_K= future_K

        self.kwargs = kwargs

        self.pipe, self.worker_pipe = multiprocessing.Pipe()
        self.start()

    def run(self):
        # initialize
        self.env = self.make(self.env_name)
        self.agent = AsyncDDPGAgent(self.env.action_space, self.env.observation_space, **self.kwargs)
        self.action_max = self.agent.action_max
        self.her_module = HERSampler('future', self.future_K, self.env.compute_reward)
        self.buffer = ReplayBuffer(self.env, self.T, self.replay_buffer_size, self.her_module.sample_her_transitions)

        while True:
            op, data = self.worker_pipe.recv()
            if op == self.EXIT:
                break
            elif op == self.START:
                n_rollout, max_timestep = data
                train_output = self.rollout(n_rollout, max_timestep)
                self.worker_pipe.send(train_output)
            elif op == self.ASK:
                obs = data['observation']
                goal = data['goal']
                inp = np.concatenate((obs, goal))
                action = self.agent(inp).detach().cpu().numpy()
                self.worker_pipe.send(action)
            elif op == self.GET_PARAM:
                self.worker_pipe.send(self.agent.get_params())

    def add_noise(self, pi, noise_eps, random_eps, action_max):
        action = pi.cpu().numpy().squeeze()
        # add the gaussian
        action += noise_eps * action_max * np.random.randn(*action.shape)
        action = np.clip(action, -action_max, action_max)
        # random actions...
        random_actions = np.random.uniform(low=-action_max, high=action_max, size=action.shape)
        # choose if use the random actions
        action += np.random.binomial(1, random_eps, 1)[0] * (random_actions - action)
        return action

    def rollout(self, n_rollout, max_timesteps):
        # new rollout
        mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []
        for _ in range(n_rollout):
            # reset the rollouts
            ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
            # reset the environment
            observation = self.env.reset()
            obs, ag, g = [observation[i] for i in ['observation', 'achieved_goal', 'desired_goal']]
            # start to collect samples
            for t in range(max_timesteps):
                inp = np.concatenate((obs, g))
                action = self.add_noise(self.agent(inp), self.noise_eps, self.random_eps, self.action_max)
                # feed the actions into the environment
                observation_new, _, _, info = self.env.step(action)
                obs_new, ag_new = observation_new['observation'], observation_new['achieved_goal']
                ep_obs.append(obs.copy()); ep_ag.append(ag.copy()); ep_g.append(g.copy()); ep_actions.append(action.copy())
                obs, ag = obs_new, ag_new

            ep_obs.append(obs.copy()); ep_ag.append(ag.copy())
            mb_obs.append(ep_obs); mb_ag.append(ep_ag); mb_g.append(ep_g); mb_actions.append(ep_actions)

        batch = [np.array(i) for i in [mb_obs, mb_ag, mb_g, mb_actions]]
        # store the episodes
        self.buffer.store_episode(batch)
        self._update_normalizer(batch)

        train_output = []
        for i in range(self.n_batchs):
            transitions = self.buffer.sample(self.batch_size)
            o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
            train_output.append(self.agent._update(
                np.concatenate((transitions['obs'], transitions['g']), axis=-1),
                transitions['action'],
                np.concatenate((transitions['obs_next'], transitions['g_next']), axis=-1),
                transitions['r'],
                done=0, #NOTE: in her, we don't think the game will terminate itself.
            ))
        return train_output

    def _update_normalizer(self, batch):
        transitions = self.her_module.sample_from_batch(episode_batch=batch)
        obs, g = transitions['obs'], transitions['g']
        self.agent.update_normalizer(np.concatenate((obs, g), axis=-1))




class DDPGAgent:
    def __init__(self, n, num_epoch, *args, **kwargs):
        self.workers = []
        for i in range(n):
            self.workers.append(Worker(*args, **kwargs))
        self.start(num_epoch)

    def start(self, num_epoch):
        pass

    def __del__(self):
        for i in self.workers:
            i.close()
