import argparse
import torch
import numpy as np

from robot.controller.pets.envs import make
from robot.envs.hyrule.rl_env import ArmReachWithXYZ
from robot.model.gnn.gnn_forward import GNNForwardAgent

from robot.model.arm.forward import Worker
from robot.model.arm.dataset import Dataset
from robot.model.arm.recorder import ModelRecorder


def make_robot_info(env: ArmReachWithXYZ, info):
    env = env.unwrapped

    class ARMGNNInfo:
        # pass
        n = 8
        m = 2 * 7
        inp_dim = (n, 4) # q, dq, force, dir
        oup_dim = (3, 2) # we will output the

        graph = torch.tensor(np.array([
            [0,1,2,3,4,5,6,1,2,3,4,5,6,7],
            [1,2,3,4,5,6,7,0,1,2,3,4,5,6],
        ]), dtype=torch.long)

        joint_dof_id = torch.tensor(env._actuator_dof['agent'], dtype=torch.long)
        edge_dof_id = torch.tensor(env._actuator_dof['agent'], dtype=torch.long)

        @classmethod
        def encode_obs(cls, state, action):
            # state (29,) action (10,)
            state = state[...,:-3].reshape(*state.shape[:-1], 2, 13)[..., cls.joint_dof_id]
            if len(state.shape) == 2:
                state = state.permute(1, 0) # 13 is the
            else:
                state = state.permute(0, 2, 1)

            action = action[...,None]
            edge_feature = torch.cat((state, action), dim=-1)
            edge_feature = torch.cat((edge_feature, edge_feature), dim=-2)
            dir = torch.cat((torch.zeros_like(action), torch.ones_like(action)), dim=-2)
            edge_feature = torch.cat((edge_feature, dir), dim=-1)

            node = torch.eye(cls.n, device=state.device)
            if len(state.shape) > 2:
                node = node[None,:].expand(state.shape[0], -1, -1)
            return node, edge_feature

        @classmethod
        def add_state(cls, state, node, edge):
            # we have the output edge
            # assme node is in the batch version
            delta = edge[:,:7]
            new_state = state[:].clone()
            new_state[..., cls.joint_dof_id] += edge[..., :cls.m//2, 0]
            new_state[..., cls.joint_dof_id + 13] += edge[..., :cls.m//2, 1]
            new_state[..., -3:] = node[..., -1,:]
            return new_state


        @classmethod
        def compute_reward(self, s, a, g, t):
            return info.compute_reward(s, a, g, t)

    return ARMGNNInfo()

def main():
    dataset = Dataset('/dataset/arm_with_geom')
    timestep = 50

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    env, info = make('armreach')
    state = env.reset()['observation']
    goal = env.reset()['desired_goal']
    action = env.action_space.sample()
    info = make_robot_info(env, info)



    model = GNNForwardAgent(
        0.001, info.encode_obs, info.add_state, info.compute_reward,
        info.inp_dim, info.oup_dim, info.graph,layers=3, mid_channels=2
    ).cuda()

    """
    state = torch.tensor(state, dtype=torch.float, device='cuda:0')[None,:]
    action = torch.tensor(action, dtype=torch.float, device='cuda:0')[None,:]
    goal = torch.tensor(goal, dtype=torch.float, device='cuda:0')[None,:]
    #node, edge = info.encode_obs(state, action)
    model.update(torch.stack((state, state), dim=1), action[:,None])
    model.rollout(state, action[:, None], goal)
    """

    num_train = 10000
    recorder = ModelRecorder(env, 'arm_gnn', save_model=slice(100000000, None, 1), network_loss=slice(0, None, 50),
                             evaluate=slice(num_train-1, None, num_train), save_video=1, max_timestep=timestep, eval_episodes=5)

    worker = Worker(env, model, dataset, num_train=num_train, batch_size=256, iter_num=20, horizon=15,
                    num_mutation=500, num_elite=50, recorder=recorder, use_geom=False)


    for i in range(1000):
        worker.epoch(num_train, 200, use_tqdm=True)

if __name__ == '__main__':
    main()