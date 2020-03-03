import argparse
from robot.envs.hyrule.legacy.sim3d import Sim3DV2, x2y, Pose
from robot.envs.hyrule.gameplay.optimizer import CEMOptimizer

def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    def make():
        sim = Sim3DV2()
        sim.stable(eps=0.01)
        sim.geo_constraint('agent', 'table', Pose([1.2, 0, 0.5], x2y), range=(20, 1000))
        return sim

    optimizer = CEMOptimizer(make, 1,
                          iter_num=20, num_mutation=300, num_elite=30, std=1., alpha=0., num_proc=20)

    sim = make()
    optimizer.optimize(sim)
    sim.parameters = []
    print(sim.cost())

    #a = sim.state_dict()
    #k = sim._constraints[1]
    #sim.step()
    #b = sim.state_dict()

    #print('check?', k.cost(sim, a, b))
    #print(sim._constraints)


    while True:
        sim.step()
        sim.render()



if __name__ == '__main__':
    main()
