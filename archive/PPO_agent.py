import numpy as np
import random

from model import ActorCriticNetwork, PolicyNetwork
from normalizer import MeanStdNormalizer

import torch
import torch.nn.functional as F
import torch.optim as optim

N = 20                  # nummber of parallel agents
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 1e-4               # learning rate
EPSILON = 0.2           # Clip hyperparameter for PPO
ROLLOUT_LEN = 1024      # Rollout length if use GAE
LAMDA = 0.95            # lambda-return for GAE
BETA = 0.001            # Coefficient for entropy loss
OPT_EPOCH = 5           # Number of updates using collected trajectories

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_np(t):
    return t.cpu().detach().numpy()

# taken from https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/utils/misc.py

def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]

class PPOAgent():
    def __init__(self, state_size, action_size, seed, hidden_layers, 
                 opt_epoch = OPT_EPOCH, use_gae = True, batch_size = BATCH_SIZE):
        """Initialize an PPO Agent object.

           Arguments
           ---------
           state_size (int): Dimension of state
           action_size (int): Total number of possible actions
           seed (int): Random seed
           hidden_layers (list): List of integers, each element represents for the size of a hidden layer
           opt_epoch (int): Nummber of updates using the collected trajectories
           use_gae (logic): Indicator of using GAE, True or False
        """
        self.state_size = state_size
        self.action_size = action_size
        self.state_normalizer = MeanStdNormalizer()
        self.opt_epoch = opt_epoch
        self.use_gae = use_gae
        self.batch_size = batch_size
        self.seed = random.seed(seed)

        # networks
        if use_gae:
            # self.network = ActorCriticNetwork(state_size, action_size, seed, hidden_layers).to(device)
            self.network = ActorCriticNetwork(state_size, action_size, seed, hidden_layers)
        else:
            # self.network = PolicyNetwork(state_size, action_size, seed, hidden_layers).to(device)
            self.network = PolicyNetwork(state_size, action_size, seed, hidden_layers)
        self.optimizer = optim.Adam(self.network.parameters(), lr = LR)

    # collect trajectories for a parallelized parallelEnv object
    def collect_trajectories(self, envs, brain_name, 
                             tmax=ROLLOUT_LEN, nrand=5, n_agents = N,
                             discount = GAMMA, lamda = LAMDA):
        """Collect trajectories.

           Arguments
           ---------
           envs: Environment
           brain_name: brain name of given environment
           tmax: Maximum length of collected trajectories
           nrand: Random steps performed before collecting trajectories
           n_agents: Number of parallel agents in the environment
        """
    
        # number of parallel instances
        n = n_agents

        #initialize returning lists and start the game!
        state_list = []
        reward_list = []
        log_prob_list = []
        action_list = []
        done_list = []
        prediction_list = []

        # reset environment
        env_info = envs.reset(train_mode=True)[brain_name]
    
        '''
        # perform nrand random steps
        for _ in range(nrand):
            actions = np.random.randn(n, self.action_size)
            actions = np.clip(actions, -1, 1)
            env_info = envs.step(actions)[brain_name]
        '''

        states = env_info.vector_observations
        # states = self.state_normalizer(states)
    
        for _ in range(tmax):
            # probs will only be used as the pi_old
            # no gradient propagation is needed
            # so we move it to the cpu
            #states_input = torch.tensor(states, dtype=torch.float, device=device)
            states_input = torch.tensor(states, dtype=torch.float)
            predictions = self.network(states_input)
            actions = to_np(predictions['a'])
            actions = np.clip(actions, -1, 1)
            env_info = envs.step(actions)[brain_name]
            next_states = env_info.vector_observations
            # next_states = self.state_normalizer(next_states)
            rewards = env_info.rewards
            dones = env_info.local_done
        
            # store the result
            state_list.append(states)
            reward_list.append(rewards)
            log_prob_list.append(to_np(predictions['log_pi_a']))
            action_list.append(actions)
            done_list.append(dones)
            prediction_list.append(predictions)

            states = next_states.copy()
        
            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if np.stack(dones).any():
                break

        # store one more step's prediction
        #states_input = torch.tensor(states, dtype=torch.float, device=device)
        states_input = torch.tensor(states, dtype=torch.float)
        predictions = self.network(states_input)
        prediction_list.append(predictions)

        # # return pi_theta, states, actions, rewards, probability
        #  return np.stack(log_prob_list), np.stack(state_list), np.stack(action_list), \
        #      np.stack(reward_list), np.stack(done_list), np.stack(prediction_list)

        # calculate accumulated discounted rewards and advantage values
        log_old_probs = np.stack(log_prob_list)
        states = np.stack(state_list)
        actions = np.stack(action_list)
        rewards = np.stack(reward_list)
        dones = np.stack(done_list)
        predictions = np.stack(prediction_list)

        # calculate accumulated discounted rewards and advantage functions
        if not self.use_gae:
            discount_seq = discount**np.arange(len(rewards))
            rewards_discounted = np.asarray(rewards)*discount_seq[:, np.newaxis]
    
            rewards_future = rewards_discounted[::-1].cumsum(axis = 0)[::-1]
            advantages = rewards_future.copy()
        else:
            T = log_old_probs.shape[0]
            rewards_future = np.zeros_like(log_old_probs)
            advantages = np.zeros_like(log_old_probs)
            tmp_adv = np.zeros(log_old_probs.shape[1])

            for i in reversed(range(T)):
                td_error = rewards[i, :] + discount * (1 - dones[i, :]) * to_np(predictions[i+1]['v']) - \
                    to_np(predictions[i]['v'])
                tmp_adv = tmp_adv * lamda * discount * (1 - dones[i, :]) + td_error
                advantages[i] = tmp_adv.copy()
                rewards_future[i] = tmp_adv + to_np(predictions[i]['v'])
    
        mean = np.mean(advantages)
        std = np.std(advantages) + 1.0e-10

        adv_normalized = (advantages - mean) / std

        # return
        return log_old_probs, states, actions, rewards, rewards_future, adv_normalized

    # clipped surrogate function
    # similar as -policy_loss for REINFORCE, but for PPO
    def clipped_surrogate(self, log_old_probs, states, actions, rewards_future, adv_normalized,
                          epsilon = EPSILON, beta = BETA):
        """Clipped surrogate function.

           Arguments
           ---------
           log_old_probs: Log probability of old policy, array with dim batch_size * 1
           states: States, array with dim batch_size * state_size
           actions: Actions, array with dim batch_size * action_size
           rewards_future: Accumulated discounted rewards, array with dim batch_size * 1
           adv_normalized: Advantage values, array with dim batch_size * 1
        """
    
        # convert everything into pytorch tensors and move to gpu if available
        # state_count = (states.shape[0], states.shape[1])

        '''
        log_old_probs = torch.tensor(log_old_probs.copy(), dtype=torch.float, device=device)
        adv = torch.tensor(adv_normalized.copy(), dtype=torch.float, device=device)
        rewards_future = torch.tensor(rewards_future.copy(), dtype=torch.float, device=device)
        states = torch.tensor(states.copy(), dtype=torch.float, device=device)
        actions = torch.tensor(actions.copy(), dtype=torch.float, device=device)
        '''
        log_old_probs = torch.tensor(log_old_probs.copy(), dtype=torch.float)
        adv = torch.tensor(adv_normalized.copy(), dtype=torch.float)
        rewards_future = torch.tensor(rewards_future.copy(), dtype=torch.float)
        states = torch.tensor(states.copy(), dtype=torch.float)
        actions = torch.tensor(actions.copy(), dtype=torch.float)

        # convert states to policy (or probability)
        # states_input = states.view(-1, self.state_size)
        # actions_input = actions.view(-1, self.action_size)
        new_predictions = self.network(states, actions)
        # log_new_probs = new_predictions['log_pi_a'].view(state_count)
        log_new_probs = new_predictions['log_pi_a'].view(-1, 1)
    
        # ratio for clipping
        ratio = (log_new_probs - log_old_probs).exp()

        # clipped function
        clip = torch.clamp(ratio, 1 - epsilon, 1 + epsilon)
        clipped_surrogate = torch.min(ratio * adv, clip * adv)

        # include entropy as a regularization term
        # entropy = new_predictions['entropy'].view(state_count)
        entropy = new_predictions['entropy'].view(-1, 1)

        # policy/actor loss
        policy_loss = -clipped_surrogate.mean() - beta * entropy.mean()

        # value/cirtic loss, if use GAE
        if self.use_gae:
            # value_loss = (rewards_future - new_predictions['v'].view(state_count)).pow(2).mean()
            value_loss = 1.0 * (rewards_future - new_predictions['v'].view(-1, 1)).pow(2).mean()
            loss = policy_loss + value_loss
        else:
            loss = policy_loss

        # this returns an average of all the loss entries
        return loss

    def step(self, envs, brain_name, grad_clip = 10):
        # first, collect trajectories
        log_old_probs, states, actions, rewards, rewards_future, adv_normalized = \
            self.collect_trajectories(envs, brain_name)

        # reshape the data
        log_old_probs_flat = log_old_probs.reshape(-1, 1)
        states_flat = states.reshape(-1, self.state_size)
        actions_flat = actions.reshape(-1, self.action_size)
        rewards_future_flat = rewards_future.reshape(-1, 1)
        adv_normalized_flat = adv_normalized.reshape(-1, 1)

        # update parameters using collected trajectories
        for _ in range(self.opt_epoch):
            # random sample from the collected trajectories by mini-batches
            sampler = random_sample(np.arange(states_flat.shape[0]), self.batch_size)

            # then updates parameters using the sampled mini-batch
            for batch_indices in sampler:
                self.network.train()
                L = self.clipped_surrogate(log_old_probs_flat[batch_indices], states_flat[batch_indices], 
                                           actions_flat[batch_indices], rewards_future_flat[batch_indices], 
                                           adv_normalized_flat[batch_indices])
                self.optimizer.zero_grad()
                L.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), grad_clip)
                self.optimizer.step()
                # del L

        return rewards