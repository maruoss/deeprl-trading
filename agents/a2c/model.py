import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPActorCritic(nn.Module):
    def __init__(self, ob_space, ac_space, min_var=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(np.prod(ob_space), 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )
        self.ac_head = nn.Linear(300, np.prod(ac_space) * 2)
        self.ac_head.weight.data.uniform_(-3e-3, 3e-3)
        self.ac_head.bias.data.uniform_(-3e-3, 3e-3)

        self.val_head = nn.Linear(300, 1)
        self.val_head.weight.data.uniform_(-3e-3, 3e-3)
        self.val_head.bias.data.uniform_(-3e-3, 3e-3)

        self.ac_space = ac_space
        self.min_var = min_var

    def forward(self, obs):
        out = obs.view(obs.size(0), -1)
        z = self.features(out)
        acs = self.ac_head(z)
        mus = acs[:, :np.prod(self.ac_space)].view(-1, *self.ac_space)
        sigs = acs[:, np.prod(self.ac_space):].view(-1, *self.ac_space)
        sigs = torch.sqrt(F.softplus(sigs) + self.min_var)
        vals = self.val_head(z)
        return mus, sigs, vals
