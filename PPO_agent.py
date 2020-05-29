import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from collections import namedtuple

from ActorCriticNetwork import ActorCriticNetwork

N = 20                  # nummber of parallel agents
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 1e-4               # learning rate
EPSILON = 0.2           # Clip hyperparameter for PPO
ROLLOUT_LEN = 1024      # Rollout length if use GAE
LAMDA = 0.95            # lambda-return for GAE
BETA = 0.001            # Coefficient for entropy loss
OPT_EPOCH = 5           # Number of updates using collected trajectories

trajectory = namedtuple('Trajectory', field_names=['state', 'action', 'reward', 'mask', 'log_prob', 'value'])

class Batcher():
    def __init__(self, states, actions, old_log_probs, returns, advantages):
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.returns = returns
        self.advantages = advantages
      
    def __len__(self):
        return len(self.returns)
   
    def __getitem__(self, index):
        return (self.states[index], 
                self.actions[index], 
                self.old_log_probs[index], 
                self.returns[index],
                self.advantages[index]
               )

def calculate_gae_returns(trajectories, num_agents, gamma=GAMMA, gae_tau=LAMDA):
    processed_trajectories = [None] * (len(trajectories)-1)
    gae = 0.
   
    for i in reversed(range(len(trajectories) - 1)):
        state, action, reward, mask, log_prob, value = trajectories[i] 
        next_value = trajectories[i+1].value

        delta = reward + gamma * next_value * mask - value
        gae = delta + gamma * gae_tau * mask * gae
        discounted_return = gae + value
        advantage = discounted_return - value
      
        processed_trajectories[i] = (state, action, log_prob, 
                                     discounted_return, advantage)
      
    return processed_trajectories

class PPOAgent():
    def __init__(self, state_size, action_size, hidden_layers, 
                 opt_epoch = OPT_EPOCH, batch_size = BATCH_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.opt_epoch = opt_epoch
        self.batch_size = batch_size
        self.model = ActorCriticNetwork(state_size, action_size, hidden_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr = LR)

    def act(self, env, brain_name, state):
        # Sample an action from current policy
        value, dist = self.model(torch.FloatTensor(state))
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Convert to numpy
        log_prob = log_prob.detach().cpu().numpy()
        value = value.detach().cpu().numpy()
        action = action.detach().cpu().numpy()
   
        # Act and see response
        env_info = env.step(np.clip(action, -1, 1))[brain_name]
        next_state = env_info.vector_observations
        reward = np.array(env_info.rewards).reshape(-1, 1)
        done = np.array(env_info.local_done)
        mask = (1-done).reshape(-1, 1)
   
        t = trajectory(state, action, reward, mask, log_prob, value)
        return next_state, done, t

    def collect_trajectories(self, env, brain_name):
        # reset environment
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        done = env_info.local_done

        # Collect trajectories
        episode_score = 0.
        trajectories = []

        while not any(done):
            next_state, done, t = self.act(env, brain_name, state)
            trajectories.append(t)
            episode_score += t.reward.mean()
            state = next_state
   
        # Obtain final value from terminal state and store it
        next_value, _ = self.model(torch.FloatTensor(state))
        next_value = next_value.detach().cpu().numpy()
        terminal_trajectory = trajectory(state, None, None, None, None, next_value)
        trajectories.append(terminal_trajectory)

        return trajectories, episode_score

    def update(self, trajectories):
        # calculate accumulated discounted returns and advantages
        p_trajectories = calculate_gae_returns(trajectories, num_agents=N)
        states, actions, old_log_probs, returns, advantages = \
            map(torch.FloatTensor, zip(*p_trajectories))
        advantages = (advantages - advantages.mean())  / (advantages.std() + 1e-7)

        # Divide collected trajectories into random batches
        batcher = DataLoader(
            Batcher(states, actions, old_log_probs, returns, advantages),
            batch_size = self.batch_size,
            shuffle = True)

        # training
        self.model.train()
        self.learn(batcher)

    def learn(self, batcher, epsilon_clip=EPSILON, beta=BETA, gradient_clip=10):
        for _ in range(self.opt_epoch):
            for states, actions, old_log_probs, returns, advantages in batcher:
                # Get updated values from policy
                values, dist = self.model(states)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy()

                # Calculate ratio and clip, so that learning doesn't change new policy much from old 
                ratio = (new_log_probs - old_log_probs).exp()
                clip = torch.clamp(ratio, 1 - epsilon_clip, 1 + epsilon_clip)
                clipped_surrogate = torch.min(ratio * advantages, clip * advantages)

                # Get losses
                actor_loss = -torch.mean(clipped_surrogate) - beta * entropy.mean()
                critic_loss = torch.mean(torch.square((returns - values)))
                losses = critic_loss * 1.0 + actor_loss

                # Do the optimizer step
                self.optimizer.zero_grad()
                losses.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
                self.optimizer.step()
