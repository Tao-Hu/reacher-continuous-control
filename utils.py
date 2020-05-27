import numpy as np
import torch

N = 20

def to_np(t):
    return t.cpu().detach().numpy()

# collect trajectories for a parallelized parallelEnv object
def collect_trajectories(envs, brain_name, action_size, agent, tmax=1000, nrand=5, n_agents = N):
    """Collect trajectories.

           Arguments
           ---------
           envs: Environment
           brain_name: brain name of given environment
           action_size: Dimension of action space
           agent: An agent
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
        actions = np.random.randn(n, action_size)
        actions = np.clip(actions, -1, 1)
        env_info = envs.step(actions)[brain_name]

    states = env_info.vector_observations
    
    for _ in range(tmax):
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        predictions = agent.network(states).squeeze().cpu().detach().numpy()
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
    predictions = agent.network(states).squeeze().cpu().detach().numpy()
    prediction_list.append(predictions)

    # return pi_theta, states, actions, rewards, probability
    return log_prob_list, state_list, action_list, \
        reward_list, done_list, prediction_list


# clipped surrogate function
# similar as -policy_loss for REINFORCE, but for PPO
def clipped_surrogate(agent, log_old_probs, states, actions, rewards, dones, predictions,
                      discount = 0.99, lamda = 0.95, epsilon = 0.1, beta = 0.01):
    """Clipped surrogate function.

    Arguments
    ---------
    agent: An agent
    log_old_probs: Log probability of old policy, array with dim rollout_length * number_of_workers
    states: States, array with dim rollout_length * number_of_workers * state_size
    actions: Actions, array with dim rollout_length * number_of_workers * action_size
    rewards: Rewards, array with dim rollout_length * number_of_workers
    dones: Indicator of the end of an episode, array with dim rollout_length * number_of_workers
    predictions: Outputs from agent's network, list of dictionary with length (rollout_length + 1)
    """

    # calculate returns
    discount_seq = discount**np.arange(len(rewards))
    rewards_discounted = np.asarray(rewards)*discount_seq[:,np.newaxis]
    
    rewards_future = rewards_discounted[::-1].cumsum(axis=0)[::-1]

    # calculate advantage functions
    if not agent.use_gae:
        advantages = rewards_future
    else:
        T = log_old_probs.shape[0]
        advantages = np.zeros_like(log_old_probs)
        tmp_adv = np.zeros(log_old_probs.shape[1])

        for i in reversed(range(T)):
            td_error = rewards[i, :] + discount * dones[i, :] * np.array([pred['v'] for pred in predictions[i+1, :]]) - \
                np.array([pred['v'] for pred in predictions[i, :]])
            tmp_adv = tmp_adv * lamda * discount * dones[i, :] + td_error
            advantages[i] = tmp_adv
    
    mean = np.mean(advantages, axis=1)
    std = np.std(advantages, axis=1) + 1.0e-10

    adv_normalized = (advantages - mean[:,np.newaxis])/std[:,np.newaxis]
    
    # convert everything into pytorch tensors and move to gpu if available
    log_old_probs = torch.tensor(log_old_probs, dtype=torch.float, device=device)
    adv = torch.tensor(adv_normalized, dtype=torch.float, device=device)
    rewards_future = torch.tensor(rewards_future, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_predictions = agent.network(states).squeeze().detach()
    log_new_probs = new_predictions['log_pi_a']
    
    # ratio for clipping
    ratio = (log_new_probs - log_old_probs).exp()

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*adv, clip*adv)

    # include entropy as a regularization term
    entropy = new_predictions['entropy']

    # policy/actor loss
    policy_loss = -clipped_surrogate-entropy

    # value/cirtic loss, if use GAE
    value_loss = 0.5 * (rewards_future)

    # this returns an average of all the loss entries
    return torch.mean(loss)