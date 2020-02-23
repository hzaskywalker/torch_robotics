import numpy as np
import pickle
import argparse
from robot.envs.hyrule.table_world import TableWorld, SetQF, Pose
from robot.envs.hyrule.gameplay.optimizer import CEMOptimizer
from robot.envs.hyrule.gameplay.waypoints import ArmMove


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='search', choices=['search', 'show'])
    args = parser.parse_args()

    horizon = 20

    def make():
        sim = TableWorld(objs=[100469, 103502, 101284],
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
        #sim.set_qf = SetQF(qf, 'agent')
        return sim

    if args.task == 'show':
        sim = make()
        with open('trajectory.pkl', 'rb') as f:
            state, output, end = pickle.load(f)
        while True:
            #sim.load_state_vector(state)
            #sim.set_param(output)
            for i in range(horizon):
                sim.step()
                sim.render()
                print(i, sim.gripper.pose)
        return

    optimizer = CEMOptimizer(make, horizon=horizon,
                             iter_num=50, num_mutation=200, num_elite=10, std=3., alpha=0.1, num_proc=20) # should be 200 for better performance
    sim = make()

    cost = ArmMove('agent', Pose([0.9, 0.2, 1.1]), None, 0.01, 1., 0, 0.)

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
