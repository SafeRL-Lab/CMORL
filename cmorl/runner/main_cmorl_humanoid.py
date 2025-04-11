import argparse
from itertools import count
from copy import deepcopy
import time
import gym
import scipy.optimize
import random
import torch
from torch.autograd import Variable
import math
import numpy as np

from torch import nn
from cmorl.algorithms.trpo import trpo_step, trpo_step_mo, cagrad
from cmorl.algorithms.trpo import Cagrad_upgrade, CR_MOGM # momentum
from cmorl.algorithms.models import Policy, Value
from cmorl.algorithms.replay_memory import Memory
from cmorl.utils.running_state import ZFilter
from cmorl.utils.utils import normal_entropy, normal_log_density, get_flat_params_from, set_flat_params_to, get_flat_grad_from
from cmorl.config.config import get_config
from cmorl.config.config_mujoco import get_env_mujoco_config


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')


args = get_config().parse_args()
env = get_env_mujoco_config(args)

if args.env_name == "Walker-dm":
    num_inputs = 24
    num_actions = 6
elif args.env_name == "Humanoid-dm":
    num_inputs = 67
    num_actions = 21
else:
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]

# env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
value_net2 = Value(num_inputs)
cost_net = Value(num_inputs)

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch, cost_batch, i_episode):
    costs = torch.Tensor(batch.cost)
    rewards = torch.Tensor(batch.reward)
    rewards2 = torch.Tensor(batch.reward2)

    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)

    values = value_net(Variable(states))
    values_cost = cost_net(Variable(states))
    values2 = value_net2(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    returns_cost = torch.Tensor(actions.size(0),1)
    deltas_cost = torch.Tensor(actions.size(0),1)
    advantages_cost = torch.Tensor(actions.size(0),1)

    returns2 = torch.Tensor(actions.size(0), 1)
    deltas2 = torch.Tensor(actions.size(0), 1)
    advantages2 = torch.Tensor(actions.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i] # A = Q-V
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    prev_return2 = 0
    prev_value2 = 0
    prev_advantage2 = 0
    for i in reversed(range(rewards2.size(0))):
        returns2[i] = rewards2[i] + args.gamma * prev_return2 * masks[i]
        deltas2[i] = rewards2[i] + args.gamma * prev_value2 * masks[i] - values2.data[i]
        advantages2[i] = deltas2[i] + args.gamma * args.tau * prev_advantage2 * masks[i]

        prev_return2 = returns2[i, 0]
        prev_value2 = values2.data[i, 0]
        prev_advantage2 = advantages2[i, 0]

    prev_return_cost = 0
    prev_value_cost = 0
    prev_advantage_cost = 0
    for i in reversed(range(costs.size(0))):
        returns_cost[i] = costs[i] + args.gamma * prev_return_cost * masks[i]
        deltas_cost[i] = costs[i] + args.gamma * prev_value_cost * masks[i] - values_cost.data[i]
        advantages_cost[i] = deltas_cost[i] + args.gamma * args.tau * prev_advantage_cost * masks[i]

        prev_return_cost = returns_cost[i, 0]
        prev_value_cost = values_cost.data[i, 0]
        prev_advantage_cost = advantages_cost[i, 0]
    
    targets_cost = Variable(returns_cost)
    ##
    targets = Variable(returns)
    targets2 = Variable(returns2)

    # Original code uses the same LBFGS to optimize the value loss
    def get_cost_loss(flat_params):
        set_flat_params_to(cost_net, torch.Tensor(flat_params))
        for param in cost_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        costs_ = cost_net(Variable(states))

        cost_loss = (costs_ - targets_cost).pow(2).mean()

        # weight decay
        for param in cost_net.parameters():
            cost_loss += param.pow(2).sum() * args.l2_reg
        cost_loss.backward()
        return (cost_loss.data.double().numpy(), get_flat_grad_from(cost_net).data.double().numpy())

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

        # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss2(flat_params):
        set_flat_params_to(value_net2, torch.Tensor(flat_params))
        for param in value_net2.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_2 = value_net2(Variable(states))

        value_loss2 = (values_2 - targets2).pow(2).mean()

        # weight decay
        for param in value_net2.parameters():
            value_loss2 += param.pow(2).sum() * args.l2_reg
        value_loss2.backward()
        return (value_loss2.data.double().numpy(), get_flat_grad_from(value_net2).data.double().numpy())

    if cost_batch >= -1.5 or i_episode <= 5:
        print("work!")
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
        set_flat_params_to(value_net, torch.Tensor(flat_params))

        flat_params2, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss2,
                                                                get_flat_params_from(value_net2).double().numpy(),
                                                                maxiter=25)
        set_flat_params_to(value_net2, torch.Tensor(flat_params2))

        advantages = (advantages - advantages.mean()) / advantages.std()
        advantages2 = (advantages2 - advantages2.mean()) / advantages2.std()

        action_means, action_log_stds, action_stds = policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()

        def get_loss2(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = policy_net(Variable(states))

            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss2 = -Variable(advantages2) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss2.mean()


        def get_kl():
            mean1, log_std1, std1 = policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        prev_policy_net = deepcopy(policy_net)
        prev_policy_net_data = get_flat_params_from(prev_policy_net)
        trpo_step(policy_net, get_loss, get_kl, 0.01, args.damping)
        grads1 = get_flat_params_from(policy_net) - prev_policy_net_data
        set_flat_params_to(policy_net, prev_policy_net_data)
        trpo_step(policy_net, get_loss2, get_kl, 0.01, args.damping)
        grads2 = get_flat_params_from(policy_net) - prev_policy_net_data
        final_grad = cagrad(grads1, grads2)
        set_flat_params_to(policy_net, prev_policy_net_data + final_grad)


    else:
        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_cost_loss, get_flat_params_from(cost_net).double().numpy(), maxiter=25)
        set_flat_params_to(cost_net, torch.Tensor(flat_params))

        advantages_cost = (advantages_cost - advantages_cost.mean()) / advantages_cost.std()

        action_means, action_log_stds, action_stds = policy_net(Variable(states))
        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_cost_loss(volatile=False):
            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages_cost) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()


        def get_kl():
            mean1, log_std1, std1 = policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
            return kl.sum(1, keepdim=True)

        trpo_step(policy_net, get_cost_loss, get_kl, 0.005, args.damping)

def get_observation_data_dm(observation):
    observations = observation
    get_observation = []
    for key in observations.values():
        # print(key)
        get_observation.append(key.tolist())
    observation_data = []
    for i in range(0, len(get_observation)):
        if isinstance(get_observation[i], list):
            for j in range(0, len(get_observation[i])):
                observation_data.append(get_observation[i][j])
        else:
            observation_data.append(get_observation[i])
    return observation_data

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

EPISODE_LENGTH = 1000
EPISODE_PER_BATCH = 16
Epoch = 5000000
alpha = 0.02
prev_policy_net = deepcopy(policy_net)
prev_cost_net = deepcopy(cost_net)

state_datas = []
velocity_datas = []
pos_datas = []
reward_datas = []
reward_datas2 = []
cost_datas = []
goal_vels = []
goal_vels.append(0.3)
for i_episode in count(1):
    memory = Memory()
    num_steps = 0
    reward_batch = 0
    reward_batch2 = 0
    cost_step = 0
    num_episodes = 0
    state = 0
    tic = time.perf_counter()
    state_data = []
    velocity_data = []
    pos_data = []
    reward_data = []
    reward_data2 = []
    cost_data = []
    while num_steps < EPISODE_LENGTH*EPISODE_PER_BATCH:
        if args.env_name == "Walker-dm" or args.env_name == "Humanoid-dm":
            step_type, reward, discount, observation = env.reset()
            state = get_observation_data_dm(observation)
            state = running_state(state)
            reward_sum = 0
            cost_sum = 0
            reward_sum2 = 0
            for t in range(EPISODE_LENGTH):  # Don't infinite loop while learning
                action = select_action(state)
                action = action.data[0].numpy()
                step_type, reward, discount, observation = env.step(action)
                next_state = get_observation_data_dm(observation)
                reward, cost, reward2 = reward
                reward_sum += reward
                reward_sum2 += reward2
                cost_sum += cost
                reward_data.append(reward)
                reward_data2.append(reward2)
                cost_data.append(cost)
                state_data.append(state)
                next_state = running_state(next_state)
                mask = 1
                if t == EPISODE_LENGTH - 1:
                    mask = 0
                memory.push(state, np.array([action]), mask, next_state, reward, reward2,
                            cost)
                state = next_state
        else:
            state, info = env.reset()
            state = running_state(state)
            reward_sum = 0
            cost_sum = 0
            reward_sum2 = 0
            for t in range(EPISODE_LENGTH):  # Don't infinite loop while learning
                action = select_action(state)
                action = action.data[0].numpy()
                next_state, reward, done, truncated, info = env.step(action)
                reward = info["reward1"]
                reward_sum += info["reward1"]
                reward2 = info["reward2"]
                reward_sum2 += info["reward2"]
                cost = info["cost"]
                cost_sum += info["cost"]
                velocity_data.append(info["x_velocity"])
                pos_data.append(info["x_position"])
                reward_data.append(info["reward1"])
                reward_data2.append(info["reward2"])
                cost_data.append(info["cost"])
                state_data.append(state)
                next_state = running_state(next_state)
                mask = 1
                if t == EPISODE_LENGTH - 1:
                    mask = 0
                memory.push(state, np.array([action]), mask, next_state, reward, reward2,
                            cost)
                if done or truncated:
                    break
                state = next_state

        num_steps += EPISODE_LENGTH
        num_episodes += 1
        reward_batch += reward_sum
        reward_batch2 += reward_sum2
        cost_step += cost_sum

    temp_val = min(i_episode/1000+1.5, 2.5)
    policy_net.action_log_std = nn.Parameter(-temp_val*torch.ones(1, num_actions))


    batch = memory.sample()
    update_params(batch, cost_step/num_steps, i_episode)
    if i_episode % args.log_interval == 0:
        print(f'Epoch {i_episode}\tAverage reward1 {reward_batch/num_steps:.7f}\tAverage reward2 {reward_batch2/num_steps:.7f}\t Average cost {-cost_step/num_steps:.7f}')

    if i_episode >= Epoch:
        break



