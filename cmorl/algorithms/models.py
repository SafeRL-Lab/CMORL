import torch
import torch.autograd as autograd
import torch.nn as nn

NEURON_COUNT = 64

class Policy(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, NEURON_COUNT)
        self.affine2 = nn.Linear(NEURON_COUNT, NEURON_COUNT)

        self.action_mean = nn.Linear(NEURON_COUNT, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)

        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))

        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.relu(self.affine1(x))
        x = torch.relu(self.affine2(x))

        action_mean = self.action_mean(x)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std) # todo: (change) add "* 0.6" for walker_dm

        return action_mean, action_log_std, action_std


class Value(nn.Module):
    def __init__(self, num_inputs):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, NEURON_COUNT)
        self.affine2 = nn.Linear(NEURON_COUNT, NEURON_COUNT)
        self.value_head = nn.Linear(NEURON_COUNT, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = torch.relu(self.affine1(x))
        x = torch.relu(self.affine2(x))

        state_values = self.value_head(x)
        return state_values
