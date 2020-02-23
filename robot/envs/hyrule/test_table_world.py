import numpy as np
import argparse
from robot.envs.hyrule.table_world import TableWorld, SetQF, Pose
from robot.envs.hyrule.gameplay.optimizer import CEMOptimizer
from robot.envs.hyrule.gameplay.waypoints import ArmMove


def main():
    parser = argparse.ArgumentParser()
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
        qf = np.array([sim.agent.get_qf() * 0 for _ in range(horizon)])
        sim.set_qf = SetQF(qf, 'agent')
        return sim

    optimizer = CEMOptimizer(make, 1,
                             iter_num=20, num_mutation=300, num_elite=30, std=1., alpha=0., num_proc=1)
    sim = make()

    cost = ArmMove('agent', Pose([0.9, 0.0, 1.1]), None, 0.01, 1., 0, 1.)

    state = sim.state_vector()
    output = optimizer.optimize(sim, cost)

    while True:
        sim.load_state_vector(state)
        sim.set_param(output)

        for i in range(horizon):
            sim.step()
            sim.render()




if __name__ == '__main__':
    main()
