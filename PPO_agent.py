import numpy as np
import random

from model import ActorCriticNetwork, PolicyNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

N = 20                  # nummber of parallel agents
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
LR = 5e-4               # learning rate
EPSILON = 0.1           # Clip hyperparameter for PPO
ROLLOUT_LEN = 1000      # Rollout length if use GAE
LAMDA = 0.95            # lambda-return for GAE
BETA = 0.01             # Coefficient for entropy loss
OPT_EPOCH = 5           # Number of updates using collected trajectories

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    def __init__(self, state_size, action_size, seed, 
                 hidden_layers, opt_epoch = OPT_EPOCH, use_gae = True):
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
        self.opt_epoch = opt_epoch
        self.use_gae = use_gae
        self.seed = random.seed(seed)

        # networks
        if use_gae:
            self.network = ActorCriticNetwork(state_size, action_size, seed, hidden_layers).to(device)
        else:
            self.network = PolicyNetwork(state_size, action_size, seed, hidden_layers).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr = LR)

    # collect trajectories for a parallelized parallelEnv object
    def collect_trajectories(self, envs, brain_name, 
                             tmax=ROLLOUT_LEN, nrand=5, n_agents = N):
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
    
        # perform nrand random steps
        for _ in range(nrand):
            actions = np.random.randn(n, self.action_size)
            actions = np.clip(actions, -1, 1)
            env_info = envs.step(actions)[brain_name]

        states = env_info.vector_observations
    
        for _ in range(tmax):
            # probs will only be used as the pi_old
            # no gradient propagation is needed
            # so we move it to the cpu
            states_input = torch.tensor(states, dtype=torch.float, device=device)
            predictions = self.network(states_input).squeeze().cpu().detach().numpy()
            actions = predictions['a']
            actions = np.clip(actions, -1, 1)
            env_info = envs.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
        
            # store the result
            state_list.append(states)
            reward_list.append(rewards)
            log_prob_list.append(predictions['log_pi_a'])
            action_list.append(actions)
            done_list.append(dones)
            prediction_list.append(predictions)

            states = next_states
        
            # stop if any of the trajectories is done
            # we want all the lists to be retangular
            if dones.any():
                break

        # store one more step's prediction
        states_input = torch.tensor(states, dtype=torch.float, device=device)
        predictions = self.network(states_input).squeeze().cpu().detach().numpy()
        prediction_list.append(predictions)

        # return pi_theta, states, actions, rewards, probability
        return log_prob_list, state_list, action_list, \
            reward_list, done_list, prediction_list

    # clipped surrogate function
    # similar as -policy_loss for REINFORCE, but for PPO
    def clipped_surrogate(self, log_old_probs, states, actions, rewards, dones, predictions,
                          discount = GAMMA, lamda = LAMDA, epsilon = EPSILON, beta = BETA):
        """Clipped surrogate function.

           Arguments
           ---------
           log_old_probs: Log probability of old policy, array with dim rollout_length * number_of_workers
           states: States, array with dim rollout_length * number_of_workers * state_size
           actions: Actions, array with dim rollout_length * number_of_workers * action_size
           rewards: Rewards, array with dim rollout_length * number_of_workers
           dones: Indicator of the end of an episode, array with dim rollout_length * number_of_workers
           predictions: Outputs from agent's network, list of dictionary with length (rollout_length + 1)
        """

        # calculate returns
        discount_seq = discount**np.arange(len(rewards))
        rewards_discounted = np.asarray(rewards)*discount_seq[:, np.newaxis]
    
        rewards_future = rewards_discounted[::-1].cumsum(axis = 0)[::-1]

        # calculate advantage functions
        if not self.use_gae:
            advantages = rewards_future
        else:
            T = log_old_probs.shape[0]
            advantages = np.zeros_like(log_old_probs)
            tmp_adv = np.zeros(log_old_probs.shape[1])

            for i in reversed(range(T)):
                td_error = rewards[i, :] + discount * dones[i, :] * np.array(predictions[i+1]['v']) - \
                    np.array(predictions[i]['v'])
                tmp_adv = tmp_adv * lamda * discount * dones[i, :] + td_error
                advantages[i] = tmp_adv
    
        mean = np.mean(advantages, axis = 1)
        std = np.std(advantages, axis = 1) + 1.0e-10

        adv_normalized = (advantages - mean[:, np.newaxis]) / std[:, np.newaxis]
    
        # convert everything into pytorch tensors and move to gpu if available
        log_old_probs = torch.tensor(log_old_probs, dtype=torch.float, device=device)
        adv = torch.tensor(adv_normalized, dtype=torch.float, device=device)
        rewards_future = torch.tensor(rewards_future, dtype=torch.float, device=device)
        states = torch.tensor(states, dtype=torch.float, device=device)
        actions = torch.tensor(actions, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        states_input = states.view(-1, states.shape[-1])
        actions_input = actions.view(-1, actions.shape[-1])
        new_predictions = self.network(states_input, actions_input)
        log_new_probs = new_predictions['log_pi_a'].view(states.shape[:-1])
    
        # ratio for clipping
        ratio = (log_new_probs - log_old_probs).exp()

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*adv, clip*adv)

        # include entropy as a regularization term
        entropy = new_predictions['entropy'].view(states.shape[:-1])

        # policy/actor loss
        policy_loss = -clipped_surrogate.mean() - beta * entropy.mean()

        # value/cirtic loss, if use GAE
        if self.use_gae:
            value_loss = 0.5 * (rewards_future - new_predictions['v'].view(states.shape[:-1])).pow(2).mean()
            loss = policy_loss + value_loss
        else:
            loss = policy_loss

        # this returns an average of all the loss entries
        return loss

    def step(self, envs, brain_name):
        # first, collect trajectories
        log_old_probs, states, actions, rewards, dones, predictions = \
            self.collect_trajectories(envs, brain_name)

        # update parameters using collected trajectories
        for _ in range(self.opt_epoch):
            L = self.clipped_surrogate(log_old_probs, states, actions, rewards, dones, predictions)
            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
            del L