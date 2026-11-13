"""Stage 1 models. Common interface: forward(x, mask) → logits (B, 12, 3).

x: (B, k, 78) float (data.scale_obs), mask: (B, k) bool (True = real step, False = padding at the episode start).
"""
import torch.nn as nn

from cybermetaformer.hidden_state.data import OBS_DIM, LABEL_HOSTS, N_CLASSES

N_OUT = len(LABEL_HOSTS) * N_CLASSES


class MLP(nn.Module):
    """Flattened k×78 input. Default (256, 256), dropout 0: the same layer order as baselines.mlp, hence the same
    initial weights with the same torch seed (reproduction of the original protocol)."""

    def __init__(self, k, hidden=(256, 256), dropout=0.0):
        super().__init__()
        layers, d = [], k * OBS_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, N_OUT))
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        return self.net(x.flatten(1)).view(-1, len(LABEL_HOSTS), N_CLASSES)
