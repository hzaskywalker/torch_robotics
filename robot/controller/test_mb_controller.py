from robot.controller.mb_controller import MBController

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
    exit(0)
    print('testing...')
    print(mb_controller.test(env))

    for i in range(1000):
        mb_controller.fit(env, num_train=5)
        if i % 10 == 0:
            print('testing...')
            print(mb_controller.test(env))



if __name__ == '__main__':
    test_mb_controller()
