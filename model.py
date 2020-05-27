import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers):
        """Initialize parameters and build a Actor Critic network.

           Arguments
           ---------
           state_size (int): Dimension of state
           action_size (int): Total number of possible actions
           seed (int): Random seed
           hidden_layers (list): List of integers, each element represents for the size of a hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.fc_actor = nn.Linear(hidden_layers[-1], action_size)
        self.fc_critic = nn.Linear(hidden_layers[-1], 1)
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state, action = None):
        """Forward pass through the network."""
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
        mean = F.tanh(self.fc_actor(state))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        v = self.fc_critic(state)
        return {'a': action, 'log_pi_a': log_prob, 'entropy': entropy, 'v': v}


class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layers):
        """Initialize parameters and build a policy network.

           Arguments
           ---------
           state_size (int): Dimension of state
           action_size (int): Total number of possible actions
           seed (int): Random seed
           hidden_layers (list): List of integers, each element represents for the size of a hidden layer
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], action_size)
        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state, action = None):
        """Forward pass through the network."""
        for linear in self.hidden_layers:
            state = F.relu(linear(state))
        mean = F.tanh(self.output(state))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1).unsqueeze(-1)
        entropy = dist.entropy().sum(-1).unsqueeze(-1)
        return {'a': action, 'log_pi_a': log_prob, 'entropy': entropy}
