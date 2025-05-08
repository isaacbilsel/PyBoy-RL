import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # μ and σ parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_eps", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init * mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init * mu_range)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())   # f(x) = sign(x)·√|x|

    def reset_noise(self):
        eps_in  = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_eps.copy_(eps_out.ger(eps_in))
        self.bias_eps.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_eps
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return F.linear(x, weight, bias)


class NoisyDDQN(nn.Module):
    """
    Distributional, Noisy, Dueling DDQN (51‑atom Categorical DQN)
    -------------------------------------------------------------
    - Outputs tensor (B, n_actions, 51)
    - Maintains separate online / target nets (target frozen)
    """

    NUM_ATOMS = 51

    def __init__(self, input_shape, n_actions):
        super().__init__()
        c, h, w = input_shape
        self.n_actions = n_actions

        # ---------- Feature extractor ----------
        self._feat_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=2),  # e.g. 20×16 → 10×8
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2), # 10×8 → 4×3
            nn.ReLU(),
            nn.Flatten()
        )

        # Get flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            self.n_flat = self._feat_extractor(dummy).shape[1]

        # ---------- Build online network ----------
        self.online = self._build_net()

        # ---------- Build frozen target network ----------
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad = False

    # ----------------------------------------------------
    # Helpers
    # ----------------------------------------------------
    def _build_net(self):
        value_stream = nn.Sequential(
            NoisyLinear(self.n_flat, 128),
            nn.ReLU(),
            NoisyLinear(128, self.NUM_ATOMS)  # 1 × atoms
        )

        advantage_stream = nn.Sequential(
            NoisyLinear(self.n_flat, 128),
            nn.ReLU(),
            NoisyLinear(128, self.n_actions * self.NUM_ATOMS)  # A × atoms
        )

        return nn.ModuleDict({
            "features": copy.deepcopy(self._feat_extractor),
            "value":    value_stream,
            "adv":      advantage_stream
        })

    # ----------------------------------------------------
    # Public API
    # ----------------------------------------------------
    def forward(self, x, model: str = "online"):
        """
        Args:
            x     : (B, C, H, W)
            model : "online" | "target"
        Returns:
            dist  : (B, n_actions, 51)  – unnormalised logits
        """
        if model not in ("online", "target"):
            raise ValueError("model must be 'online' or 'target'")
        net = self.online if model == "online" else self.target

        feat = net["features"](x)                       # (B, n_flat)
        val  = net["value"](feat).view(-1, 1, self.NUM_ATOMS)
        adv  = net["adv"](feat).view(-1, self.n_actions, self.NUM_ATOMS)

        # Dueling combination (only across actions!)
        q = val + (adv - adv.mean(dim=1, keepdim=True))
        return q                                         # (B, A, 51)

    def reset_noise(self):
        for m in self.online.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()
        # Target net is frozen; noise reset not needed there.