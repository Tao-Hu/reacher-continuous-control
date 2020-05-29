import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_layers):
       super(ActorCriticNetwork, self).__init__()
       self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

       self.std = nn.Parameter(torch.zeros(1, action_size))

       self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
       layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
       self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

       self.actor_lin = nn.Linear(hidden_layers[-1], action_size)
       self.critic_lin = nn.Linear(hidden_layers[-1], 1)

    def forward(self, state):
       x = state.clone()
       for linear in self.hidden_layers:
          x = F.relu(linear(x))

       # Critic
       value = self.critic_lin(x)

       # Actor
       actor_output = self.actor_lin(x)
       mean = torch.tanh(actor_output)
       dist = Normal(mean, F.softplus(self.std))
      
       return value, dist