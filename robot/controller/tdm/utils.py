import numpy as np
import abc


def truncated_geometric(p, truncate_threshold, size, new_value=None):
    """
    Sample from geometric, but truncated values to `truncated_threshold`.
    All values greater than `truncated_threshold` will be set to `new_value`.
    If `new_value` is None, then they will be assigned random integers from 0 to
    `truncate_threshold`.
    :param p: probability parameter for geometric distribution
    :param truncate_threshold: Cut-off
    :param size: size of sample
    :param new_value:
    :return:
    """
    samples = np.random.geometric(p, size)
    samples_too_large = samples > truncate_threshold
    num_bad = sum(samples_too_large)
    if new_value is None:
        samples[samples > truncate_threshold] = (
            np.random.randint(0, truncate_threshold, num_bad)
        )
    else:
        samples[samples > truncate_threshold] = new_value
    return samples


class ScalarSchedule(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def get_value(self, t):
        pass


class ConstantSchedule(ScalarSchedule):
    def __init__(self, value):
        self._value = value

    def get_value(self, t):
        return self._value


def _expand_goal(goal, path_length):
    return np.repeat(
        np.expand_dims(goal, 0),
        path_length,
        0,
    )


def tdm_rollout(
    env,
    agent,
    qf=None,
    vf=None,
    max_path_length=np.inf,
    animated=False,
    init_tau=0.0,
    decrement_tau=False,
    cycle_tau=False,
    get_action_kwargs=None,
    observation_key=None,
    desired_goal_key=None,
    vis_list=list(),
    dont_terminate=False,
    epoch=None,
    rollout_num=None,
):
    full_observations = []
    if get_action_kwargs is None:
        get_action_kwargs = {}
    observations = []
    next_observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    taus = []
    agent.reset()
    path_length = 0
    if animated:
        env.render()

    tau = np.array([init_tau])
    o = env.reset()

    goal = env.get_goal()
    agent_goal = goal
    if desired_goal_key:
        agent_goal = agent_goal[desired_goal_key]

    while path_length < max_path_length:
        full_observations.append(o)
        agent_o = o
        if observation_key:
            agent_o = agent_o[observation_key]

        a, agent_info = agent.get_action(agent_o, agent_goal, tau, **get_action_kwargs)
        if animated:
            env.render()
        if hasattr(env, 'set_tau'):
            if 'tau' in agent_info:
                env.set_tau(agent_info['tau'])
            else:
                env.set_tau(tau)
        next_o, r, d, env_info = env.step(a)

        if 'latent' in observation_key:
            key = 'image_observation'
        else:
            key = observation_key

        update_next_obs(
            next_o, o, key, env, agent, qf, vf, vis_list, epoch, rollout_num, path_length,
            agent_info.get('tau', tau),
        )

        next_observations.append(next_o)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        taus.append(tau.copy())
        path_length += 1
        if decrement_tau:
            tau -= 1
        if tau < 0:
            if cycle_tau:
                if init_tau > max_path_length - path_length - 1:
                    tau = np.array([max_path_length - path_length - 1])
                else:
                    tau = np.array([init_tau])
            else:
                tau = np.array([0])
        if d and not dont_terminate:
            break
        o = next_o
    full_observations.append(o)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    next_observations = np.array(next_observations)

    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=np.array(agent_infos),
        env_infos=np.array(env_infos),
        num_steps_left=np.array(taus),
        goals=_expand_goal(agent_goal, len(terminals)),
        full_observations=full_observations,
    )



class MultigoalSimplePathSampler(object):
    def __init__(
            self,
            env,
            policy,
            max_samples,
            max_path_length,
            tau_sampling_function,
            qf=None,
            cycle_taus_for_rollout=True,
            render=False,
            observation_key=None,
            desired_goal_key=None,
    ):
        self.env = env
        self.policy = policy
        self.qf = qf
        self.max_samples = max_samples
        self.max_path_length = max_path_length
        self.tau_sampling_function = tau_sampling_function
        self.cycle_taus_for_rollout = cycle_taus_for_rollout
        self.render = render
        self.observation_key = observation_key
        self.desired_goal_key = desired_goal_key

    def obtain_samples(self):
        paths = []
        for i in range(self.max_samples // self.max_path_length):
            tau = self.tau_sampling_function()
            path = multitask_rollout(
                self.env,
                self.policy,
                self.qf,
                init_tau=tau,
                max_path_length=self.max_path_length,
                decrement_tau=self.cycle_taus_for_rollout,
                cycle_tau=self.cycle_taus_for_rollout,
                animated=self.render,
                observation_key=self.observation_key,
                desired_goal_key=self.desired_goal_key,
            )
            paths.append(path)
        return paths


def multitask_rollout(*args, **kwargs):
    # TODO Steven: remove pointer
    return tdm_rollout(*args, **kwargs)


import torch
_use_gpu = True
def from_numpy(*args, **kwargs):
    if _use_gpu:
        return torch.from_numpy(*args, **kwargs).float().cuda()
    else:
        return torch.from_numpy(*args, **kwargs).float()

def _elem_or_tuple_to_variable(elem_or_tuple):
    if isinstance(elem_or_tuple, tuple):
        return tuple(
            _elem_or_tuple_to_variable(e) for e in elem_or_tuple
        )
    return from_numpy(elem_or_tuple).float()


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch):
    return {
        k: _elem_or_tuple_to_variable(x)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }