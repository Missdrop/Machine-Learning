import torch


class Tool:
    def soft_update(_from, _to):
        for _from, _to in zip(_from.parameters(), _to.parameters()):
            value = _to.data * 0.99 + _from.data * 0.01
            _to.data.copy_(value)

    def requires_grad(model, value):
        for param in model.parameters():
            param.requires_grad_(value)

    def get_action_entropy(mu, sigma):
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.rsample()
        entropy = dist.entropy()
        return action, entropy
