import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

#based on https://colab.research.google.com/github/leox1v/dl20/blob/master/Transformers_Solution.ipynb#scrollTo=7JaF6C3Dfdog&uniqifier=2 


class SelfAttention(nn.Module):
    """
    A SelfAttention model.
    
    Args:
        d: The embedding dimension.
        heads: The number of attention heads.
    """
    def __init__(self, d: int, heads: int=8) -> None:   
        super().__init__()
        self.h = heads

        self.Wq = nn.Linear(d, d * heads, bias=False)
        self.Wk = nn.Linear(d, d * heads, bias=False)
        self.Wv = nn.Linear(d, d * heads, bias=False)

        # Unifying outputs
        self.unifyheads = nn.Linear(heads * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input embedding of shape [b, l, d]
        
        Returns:
            Self attention tensor of shape [b, l, d].
        """
        # b = batch
        # l = sequence length
        # d = embedding dim.

        b, l, d = x.size()
        h = self.h

        # Transform input embeddings of shape [b, l, d] to queries, keys and values.
        # The output shape is [b, l, d*h] which we transform into [b, l, h, d]. Then,
        # we fold the heads into the batch dimension to arrive at [b*h, l, d].
        queries = self.Wq(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        keys = self.Wk(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)
        values = self.Wv(x).view(b, l, h, d).transpose(1, 2).contiguous().view(b*h, l, d)

        # Compute raw weights of shape [b*h, l, l]
        w_prime = torch.bmm(queries, keys.transpose(1, 2)) / np.sqrt(d)

        # Compute normalized weights by normalizing last dim. 
        # Shape: [b*h, l, l]
        w = F.softmax(w_prime, dim=-1)

        # Apply self attention to the values
        # Shape [b, h, l, d]
        out = torch.bmm(w, values).view(b, h, l, d)

        # Swap h, l back
        out = out.transpose(1, 2).contiguous().view(b, l, h * d)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    """
    A Transformer block consisting of self attention and ff-layers.

    Args:
        d (int): The embedding dimension
        heads (int): The number of attention heads
        n_mlp (int): n_mlp * d = hidden dim of independent FFNs
    """
    def __init__(self, d: int, heads: int=8, n_mlp: int=4) -> None:
        super().__init__()

        # The self-attention layer
        self.attention = SelfAttention(d=d, heads=heads)

        # The two layer norms
        self.norm1 = nn.LayerNorm(d)
        self.norm2 = nn.LayerNorm(d)

        # The feed-forward layer
        self.ff = nn.Sequential(
            nn.Linear(d, d * n_mlp),
            nn.ReLU(),
            nn.Linear(d * n_mlp, d)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The input sequence embedding of shape [b, l, d]

        Returns:
            Transformer output tensor of shape [b, l, d]
        """
        out = self.attention(x)
        out = self.norm1(out + x)
        out = self.ff(out) + out
        out = self.norm2(out)

        return out


class MLPActor(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()

        # Stack of 6 Transformer blocks as in original implementation
        self.transformer_encoder = nn.Sequential(
            TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
        )
        # To go from shape [b, l, d] to [b, l, 1]
        self.linearhead = nn.Linear(ob_space[1], 1)

        self.ac_head = nn.Linear(ob_space[0], np.prod(ac_space))
        self.ac_head.weight.data.uniform_(-3e-3, 3e-3)
        self.ac_head.bias.data.uniform_(-3e-3, 3e-3)
        self.ac_space = ac_space


    def forward(self, obs):
        out = self.transformer_encoder(obs)
        out = self.linearhead(out)
        out = out.squeeze(dim=-1)
        out = self.ac_head(out)
        out = out.view(obs.size(0), *self.ac_space)
        # nn.functional.softmax(out, dim=-1)
        return out


class MLPCritic(nn.Module):
    def __init__(self, ob_space, ac_space):
        super().__init__()

        # Stack of 6 Transformer blocks as in original implementation
        self.transformer_encoder = nn.Sequential(
            TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
            # TransformerBlock(d=ob_space[1]),
        )
        # self.transformerblock = TransformerBlock(d=ob_space[1])
        # self.flatten = nn.Flatten()
        self.linearhead = nn.Linear(ob_space[1], 300)

        self.acs_features = nn.Sequential(
            nn.Linear(np.prod(ac_space), 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
        )

        # connect transformer output (state) and ffn output (action)
        self.val_head = nn.Linear(300 + 300, 1)
        self.val_head.weight.data.uniform_(-3e-3, 3e-3)
        self.val_head.bias.data.uniform_(-3e-3, 3e-3)

    def forward(self, obs, acs):
        obs = self.transformer_encoder(obs)
        # Either flatten:
        # obs = self.flatten(obs)
        # or take mean of sequences (less params.)
        obs = obs.mean(dim=1)
        obs = self.linearhead(obs)
        acs = self.acs_features(acs)
        out = torch.cat([obs, acs], dim=-1)
        out = self.val_head(out)
        return out
