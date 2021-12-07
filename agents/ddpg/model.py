import torch
import torch.nn as nn
import numpy as np


class MLPActor(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(np.prod(ob_space), 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
        self.ac_head = nn.Linear(300, np.prod(ac_space))
        self.ac_head.weight.data.uniform_(-3e-3, 3e-3)
        self.ac_head.bias.data.uniform_(-3e-3, 3e-3) #TODO: fan-in initialization for other layers?
        self.ac_space = ac_space


    def forward(self, obs):
        out = obs.view(obs.size(0), -1)
        out = self.features(out)
        out = self.ac_head(out)
        out = out.view(obs.size(0), *self.ac_space)
        return out


class MLPCritic(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(np.prod(ob_space) + np.prod(ac_space), 400), #TODO: Action only come in, in the second layer?
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
        self.val_head = nn.Linear(300, 1)
        self.val_head.weight.data.uniform_(-3e-3, 3e-3)
        self.val_head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, acs):
        obs = obs.view(obs.size(0), -1)
        acs = acs.view(acs.size(0), -1)
        out = torch.cat([obs, acs], dim=-1)
        out = self.features(out)
        out = self.val_head(out)
        return out
