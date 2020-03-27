from robot.utils.trainer import AgentBase

class RolloutAgent(AgentBase):
    def __init__(self, model, lr, loss_weights):
        self.model = model
        super(RolloutAgent, self).__init__(model, lr)
        self.loss_weights = loss_weights

    def rollout(self, s, a, goal=None):
        # s (inp_dim)
        # a (pop, T, acts)
        predict, reward = [], 0

        for i in range(a.shape[1]):
            ai = a[:, i]
            t = s.add(*self.model(*s.as_input(ai)))
            predict.append(t)

            if goal is not None:
                reward = t.compute_reward(s, ai, goal) + reward
            s = t

        return s.stack(predict), reward

    def update(self, state, actions, future):
        # state is the frame
        if self.training:
            self.optim.zero_grad()

        predict, _ = self.rollout(state, actions, None)
        losses = predict.calc_loss(future)

        try:
            losses['model_decay'] = self.model.loss()
        except AttributeError as e:
            pass

        total_loss = 0
        assert len(losses) == len(self.loss_weights), "Please assign weights for all losses"

        for name, value in self.loss_weights.items():
            assert name in losses
            total_loss = losses[name] * value + total_loss
            losses[name] = losses[name].detach().cpu().numpy()

        if self.training:
            total_loss.backward()
            self.optim.step()

        return {'predict': predict.cpu(), **losses}

    def update_normalizer(self, batch):
        pass
