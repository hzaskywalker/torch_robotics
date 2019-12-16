def discount_with_dones(rewards, last_value, dones, gamma, use_gae=False, values=None, tau=0.95):
    n = rewards.size()[1]
    discounted = rewards * 0
    if values is not None:
        assert rewards.shape == values.shape

    if use_gae:
        gae = 0
        next_value = last_value
        for i in range(n-1, -1, -1):
            assert rewards[:,i].shape == values[:,i].shape and values[:,i].shape == dones[:,i].shape
            delta = rewards[:, i] + gamma * next_value * (1-dones[:, i]) - values[:, i]
            gae = delta + gamma * tau * (1-dones[:, i]) * gae
            discounted[:,i] = gae + values[:, i]
            next_value = values[:,i]
        return discounted

    #raise NotImplementedError
    r = last_value
    if values is not None:
        assert rewards.shape == dones.shape and dones[:,0].shape == r.shape
    for i in range(n-1, -1, -1):
        r = rewards[:,i] + (1-dones[:,i]) * gamma * r
        discounted[:,i] = r
    return discounted