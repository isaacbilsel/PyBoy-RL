import torch.nn as nn
import copy
import torch
import math

# based on pytorch tutorial by yfeng997: https://github.com/yfeng997/MadMario/blob/master/neural.py

class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim
    
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3744, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
           p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

class DuelDDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()

        c, h, w = input_shape

        # Adjusted Conv layers for small input
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2),  # 20x16 → 10x8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # 10x8 → 4x3
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            n_flatten = self.feature_extractor(dummy_input).shape[1]

        # Dueling streams
        self.value_stream = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(n_flatten, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

        self.online = self._build_net()
        self.target = self._build_net()

    def _build_net(self):
        return nn.ModuleDict({
            "features": self.feature_extractor,
            "value": self.value_stream,
            "advantage": self.advantage_stream
        })

    def forward(self, x, model):
        net = self.online if model == "online" else self.target

        features = net["features"](x)
        value = net["value"](features)
        advantage = net["advantage"](features)

        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals


class NoisyDDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.action_space_dim = n_actions


        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            n_flatten = self.feature_extractor(dummy_input).shape[1]

        self.value_stream = nn.Sequential(
            NoisyLinear(n_flatten, 128),
            nn.ReLU(),
            NoisyLinear(128, 51)  # <--- output 51 atoms for value
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(n_flatten, 128),
            nn.ReLU(),
            NoisyLinear(128, n_actions * 51)  # <--- output 51 atoms for each action
        )


        self.online = self._build_net()
        self.target = self._build_net()

    def _build_net(self):
        return nn.ModuleDict({
            "features": self.feature_extractor,
            "value": self.value_stream,
            "advantage": self.advantage_stream
        })

    def forward(self, x, model):
        net = self.online if model == "online" else self.target
        features = net["features"](x)

        value = net["value"](features)  # shape (batch, 51)
        advantage = net["advantage"](features)  # shape (batch, n_actions * 51)

        # Reshape
        batch_size = x.size(0)
        value = value.view(batch_size, 1, 51)
        advantage = advantage.view(batch_size, self.action_space_dim, 51)

        advantage = advantage - advantage.mean(dim=2, keepdim=True)

        # Combine dueling streams
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return qvals

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(input, weight, bias)