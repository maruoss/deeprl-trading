import torch
import torch.nn as nn
import numpy as np


class MLPActor(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(ob_space[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
            # nn.Conv1d(128, 128, 5),
            # nn.ReLU(),
            nn.Flatten()
        )

        # get size for linear layer
        sample = torch.zeros(1, *ob_space)
        size = self.features(sample).shape

        self.ac_head = nn.Linear(size[1], np.prod(ac_space))
        self.ac_head.weight.data.uniform_(-3e-3, 3e-3)
        self.ac_head.bias.data.uniform_(-3e-3, 3e-3)
        self.ac_space = ac_space


    def forward(self, obs):
        # out = obs.view(obs.size(0), -1)
        out = self.features(obs)
        out = self.ac_head(out)
        out = out.view(obs.size(0), *self.ac_space)
        # nn.functional.softmax(out, dim=-1)
        return out


class MLPCritic(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(ob_space[0], 128, 5),
            nn.ReLU(),
            nn.Conv1d(128, 128, 5),
            nn.ReLU(),
            # nn.Conv1d(128, 128, 5),
            # nn.ReLU(),
            nn.Flatten()
        )
        sample = torch.zeros(1, *ob_space)
        size = self.features(sample).shape

        self.acs_features = nn.Sequential(
            nn.Linear(np.prod(ac_space), 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )

        self.val_head = nn.Linear(size[1] + 300, 1)
        self.val_head.weight.data.uniform_(-3e-3, 3e-3)
        self.val_head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, acs):
        obs = self.features(obs)
        acs = self.acs_features(acs)
        out = torch.cat([obs, acs], dim=-1)
        out = self.val_head(out)
        return out
