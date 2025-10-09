import torch

class Actor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.network(state)

class Cretic(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Tanh(),
        )

    def forward(self, state):
        return self.network(state)

class SACActor(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.s = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Tanh(),
        )
        self.sigma = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Tanh(),
        )

    def forward(self, state):
        state = self.s(state)
        return self.mu(state), self.sigma(state).exp()