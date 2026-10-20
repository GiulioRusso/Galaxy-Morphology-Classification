"""Custom losses. Any torch.nn loss works by name (`loss: {name: CrossEntropyLoss}`); a class defined here is
selected the same way. `weight: balanced` is replaced by inverse class frequencies of the training split.
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, 1).gather(1, targets[:, None]).squeeze(1)
        return (-(1 - logp.exp()) ** self.gamma * logp).mean()
