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
            nn.Linear(300, np.prod(ac_space)),
        )
        self.ac_space = ac_space

    def forward(self, obs):
        out = obs.view(obs.size(0), -1)
        out = self.features(out)
        out = out.view(obs.size(0), *self.ac_space)
        return out


class MLPCritic(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(np.prod(ob_space) + np.prod(ac_space), 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, obs, acs):
        obs = obs.view(obs.size(0), -1)
        acs = acs.view(acs.size(0), -1)
        out = torch.cat([obs, acs], dim=-1)
        out = self.features(out)
        return out