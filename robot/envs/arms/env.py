import gym
import importlib
import numpy as np

class Env(gym.Env):
    def __init__(self, name, dt=1e-2):
        super(Env, self).__init__()

        if name[:4] == 'arm1':
            subfolder = 'one_link'
        if name[:4] == 'arm2':
            subfolder = 'two_link'
        elif name[:4] == 'arm3':
            subfolder = 'three_link'
        else:
            raise NotImplementedError

        arm_name = 'robot.envs.arms.%s.%s' % (subfolder, 'arm' + name[4:])
        arm_module = importlib.import_module(name=arm_name)

        self.dt = dt
        self.arm = arm_module.Arm(dt=dt)
        self.dof = self.arm.DOF

        # suppose there is no encoding....

        high = np.array([np.pi, np.pi, np.pi, 1.2, 1.2, 1.2])
        low = -high
        self.observation_space = gym.spaces.Box(low=low, high=high)

        action = np.array([50, 50, 50])
        self.action_space = gym.spaces.Box(low=-action, high=action)

        self.state_dim = 3 + 3 + 6 + 6
        self.action_dim = 1

    def reset(self):
        self.arm.reset()
        xnext = np.hstack([np.copy(self.arm.q),
                           np.copy(self.arm.dq)])
        self.cur = xnext.copy()
        return xnext

    def step(self, u):
        self.cur = self.plant_dynamics(self.cur, u)
        return self.cur.copy(), 0, False, {}

    def gen_target(self):
        out = np.random.random(size=(2,)) * np.sum(self.arm.L) * .75
        assert (out ** 2).sum() < np.sum(self.arm.L) ** 2
        return out

    def cost(self, x, u):
        """ the immediate state cost function """
        # compute cost
        dof = u.shape[0]
        num_states = x.shape[0]

        l = np.sum(u**2)

        # compute derivatives of cost
        l_x = np.zeros(num_states)
        l_xx = np.zeros((num_states, num_states))
        l_u = 2 * u
        l_uu = 2 * np.eye(dof)
        l_ux = np.zeros((dof, num_states))

        # returned in an array for easy multiplication by time step
        #return l, l_x, l_xx, l_u, l_uu, l_ux
        return l * self.dt, np.concatenate((l_x, l_u)) * self.dt, np.vstack([
            np.hstack(
                [l_xx, l_ux.T],
            ),
            np.hstack(
                [l_ux, l_uu]
            )
        ]) * self.dt


    def cost_final(self, x, target):
        """ the final state cost function """
        num_states = x.shape[0]
        l_x = np.zeros((num_states + self.arm.DOF))
        l_xx = np.zeros((num_states + self.arm.DOF, num_states + self.arm.DOF))

        wp = 1e4 # terminal position cost weight
        wv = 1e4 # terminal velocity cost weight

        xy = self.arm.x
        xy_err = np.array([xy[0] - target[0], xy[1] - target[1]])
        l = (wp * np.sum(xy_err**2) +
                wv * np.sum(x[self.arm.DOF:self.arm.DOF*2]**2))

        l_x[0:self.arm.DOF] = wp * self.dif_end(x[0:self.arm.DOF], target)
        l_x[self.arm.DOF:self.arm.DOF*2] = (2 *
                wv * x[self.arm.DOF:self.arm.DOF*2])

        eps = 1e-4 # finite difference epsilon
        # calculate second derivative with finite differences
        for k in range(self.arm.DOF):
            veps = np.zeros(self.arm.DOF)
            veps[k] = eps
            d1 = wp * self.dif_end(x[0:self.arm.DOF] + veps, target)
            d2 = wp * self.dif_end(x[0:self.arm.DOF] - veps, target)
            l_xx[0:self.arm.DOF, k] = ((d1-d2) / 2.0 / eps).flatten()

        l_xx[self.arm.DOF:self.arm.DOF*2, self.arm.DOF:self.arm.DOF*2] = 2 * wv * np.eye(self.arm.DOF)

        # Final cost only requires these three values
        return l, l_x, l_xx


    # Compute derivative of endpoint error
    def dif_end(self, x, target):

        xe = -target.copy()
        for ii in range(self.arm.DOF):
            xe[0] += self.arm.L[ii] * np.cos(np.sum(x[:ii+1]))
            xe[1] += self.arm.L[ii] * np.sin(np.sum(x[:ii+1]))

        edot = np.zeros((self.arm.DOF,1))
        for ii in range(self.arm.DOF):
            edot[ii,0] += (2 * self.arm.L[ii] *
                    (xe[0] * -np.sin(np.sum(x[:ii+1])) +
                     xe[1] * np.cos(np.sum(x[:ii+1]))))
        edot = np.cumsum(edot[::-1])[::-1][:]

        return edot


    def finite_differences(self, x, u):
        """ calculate gradient of plant dynamics using finite differences
        x np.array: the state of the system
        u np.array: the control signal
        """
        dof = u.shape[0]
        num_states = x.shape[0]

        A = np.zeros((num_states, num_states))
        B = np.zeros((num_states, dof))

        eps = 1e-4 # finite differences epsilon
        for ii in range(num_states):
            # calculate partial differential w.r.t. x
            inc_x = x.copy()
            inc_x[ii] += eps
            state_inc = self.plant_dynamics(inc_x, u.copy())
            dec_x = x.copy()
            dec_x[ii] -= eps
            state_dec = self.plant_dynamics(dec_x, u.copy())
            A[:, ii] = (state_inc - state_dec) / (2 * eps)

        for ii in range(dof):
            # calculate partial differential w.r.t. u
            inc_u = u.copy()
            inc_u[ii] += eps
            state_inc = self.plant_dynamics(x.copy(), inc_u)
            dec_u = u.copy()
            dec_u[ii] -= eps
            state_dec = self.plant_dynamics(x.copy(), dec_u)
            B[:, ii] = (state_inc - state_dec) / (2 * eps)

        return A, B


    def plant_dynamics(self, x, u):
        """ simulate a single time step of the plant, from
        initial state x and applying control signal u
        x np.array: the state of the system
        u np.array: the control signal
        """

        # set the arm position to x
        self.arm.reset(q=x[:self.arm.DOF],
                       dq=x[self.arm.DOF:self.arm.DOF * 2])

        # apply the control signal
        self.arm.apply_torque(u, self.arm.dt)
        # get the system state from the arm
        xnext = np.hstack([np.copy(self.arm.q),
                           np.copy(self.arm.dq)])
        return xnext


    def forward(self, x, u):
        return self.plant_dynamics(x, u)

    def get_graph(self):
        return np.array(
            [
                [0, 1, 2],
                [1, 2, 3],
            ]
        )

    def get_node_attr(self):
        # since all links have the same mass, we
        return np.array(
            [
                [1, 0], # ground
                [0, 1], # object
                [0, 1], # object
                [0, 1], # object
            ]
        )

    def get_edge_attr(self):
        return np.array(
            [
                [0],  # the edge are the same
                [0],  # the edge are the same
                [0],  # the edge are the same
            ]
        )

    def encode(self, x, dt=1e-4):
        # return the x, w, dx and dw
        w1 = x[:self.dof].copy()
        p1 = self.arm.position(w1)

        w2 = x[:self.dof] + x[self.dof:self.dof*2] * dt
        p2 = self.arm.position(w2)

        dw = (w2 - w1)/dt

        state = np.zeros((p1.shape[1], 18))
        state[:, :2] = p1.T
        state[:, 3:5] = ((p2-p1)/dt).T

        state[1:, 6:6+4] = np.stack((
            np.cos(w1), -np.sin(w1),
            np.sin(w1), np.cos(w1),
        ), axis=1)

        state[1:, 12:12+4] = np.stack((
            np.cos(dw), -np.sin(dw),
            np.sin(dw), np.cos(dw),
        ), axis=1)

        return state


if __name__ == '__main__':
    arm = Env('arm2')
