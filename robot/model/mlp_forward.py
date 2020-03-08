import torch
from robot.utils import AgentBase, tocpu
from robot.model.gnn.gnn_forward import mlp, Concat

class MLPForward(AgentBase):
    def __init__(self, lr, env, *args, **kwargs):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        model = mlp(state_dim + action_dim, state_dim, *args, **kwargs)
        model = Concat(model)

        super(MLPForward, self).__init__(model, lr)

        self.forward_model = model
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optim, 0.975)
        self.step = 0

    def get_predict(self, s, a, update=True):
        output = self.forward_model(s, a)
        if update:
            return s + output
        else:
            return output

    def __call__(self, s, a):
        return s + self.forward_model(s, a)

    def update(self, s, a, t):
        # predict t given s, and a
        delta = t - s

        if self.training:
            self.optim.zero_grad()

        output = self.get_predict(s, a, update=False)
        loss = torch.nn.functional.mse_loss(output, delta)

        if self.training:
            loss.backward()
            self.optim.step()

            self.step += 1
            if self.step % 50000 == 0:
                self.lr_scheduler.step(self.step//50000)
        return {
            'loss': tocpu(loss)
        }

