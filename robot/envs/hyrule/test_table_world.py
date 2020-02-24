import numpy as np
import pickle
import argparse
from robot.envs.hyrule.table_world import TableWorld, SetQF, Pose
from robot.envs.hyrule.gameplay.optimizer import CEMOptimizer
from robot.envs.hyrule.gameplay.waypoints import ArmMove, ObjectMove, Grasped, WaypointList, ControlNorm, Trajectory


import gym
class Env(gym.Env):
    # xvfb-run python3 trajopt.py --env_name table_world --horizon 10 --iter_num 10 --num_mutation 200 --timestep 100

    def __init__(self):
        super(Env, self).__init__()
        sim = TableWorld([], None)
        for i in range(100):
            sim.step()
        self.sim = sim

        low = self.state_vector()*0-np.inf
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=low.shape, dtype=np.float32,
        )
        act_shape = np.array(self.sim.agent.get_qf()).shape
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=act_shape, dtype=np.float32,
        )
        for i in range(100):
            self.sim.step()
        self.sim.timestep = 0

        self.cost = Trajectory(
            (WaypointList(
                ObjectMove('box', Pose([0.9, 0.2, 1.]), 2., 0.),
                Grasped('agent', 'box', 1),
                ControlNorm('agent', 0.0001)
            ), 50),
            (WaypointList(
                ObjectMove('box', Pose([0.8, -0.2, 0.55]), 2, 0),
                Grasped('agent', 'box', 1),
                ControlNorm('agent', 0.0001)
            ), 50)
        )


    def state_vector(self):
        return self.sim.state_vector()

    def set_state(self, qpos, qvel):
        self.sim.load_state_vector(np.concatenate((qpos, qvel)))

    def reset(self):
        self.sim.timestep = 0
        return self.sim.state_vector()

    def step(self, action):
        self.sim.agent.set_qf(action * 10)
        for i in range(1):
            self.sim.step()
        return self.state_vector(), -self.cost.cost(self.sim), False, {}

    def render(self, mode='human'):
        return self.sim.render(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='search', choices=['search', 'show'])
    args = parser.parse_args()

    horizon = 100

    def make():
        sim = TableWorld(objs=[], #100469, 103502, 101284
                         map=np.array([
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0],
                             [0, 0, 0, 2, 0],
                             [0, 3, 0, 0, 0],
                         ]),
                         dir=None,
                         names=['bucket', 'haha2', 'glass'])
        for i in range(100):
            sim.step()
        qf = np.array([sim.agent.get_qf() * 0 for _ in range(horizon)])
        sim.set_qf = SetQF(qf, 'agent')
        return sim

    if args.task == 'show':
        sim = make()
        with open('trajectory.pkl', 'rb') as f:
            state, output, end = pickle.load(f)
        while True:
            sim.load_state_vector(state)
            sim.set_param(output)
            for i in range(horizon):
                sim.step()
                sim.render()
                #print(i, sim.gripper.pose)
        return

    optimizer = CEMOptimizer(make, horizon=horizon,
                             iter_num=50, num_mutation=200, num_elite=10, std=3., alpha=0.1, num_proc=20) # should be 200 for better performance
    sim = make()

    #cost = ArmMove('agent', Pose([0.9, 0.2, 1.1]), None, 0.01, 1., 0, 0.)
    cost = WaypointList(
        ObjectMove('box', Pose([0.9, 0.2, 1.]), 2., 0.),
        Grasped('agent', 'box', 1)
    )

    state = sim.state_vector()
    output = optimizer.optimize(sim, cost, show_progress=True)

    while True:
        sim.load_state_vector(state)
        sim.set_param(output)

        for i in range(horizon):
            sim.step()
            sim.render()
            print(i, sim.gripper.pose)

        with open('trajectory.pkl', 'wb') as f:
            pickle.dump([state, output, sim.state_vector()], f)
        return


if __name__ == '__main__':
    main()
