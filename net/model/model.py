"""Model: any timm backbone with a fresh head. Arguments are the keys under `model:` in the config.

Zoobot encoders load the same way: name="hf_hub:mwalmsley/zoobot-encoder-convnext_tiny".
in_chans=4 adds the segmentation channel (timm adapts the pretrained first convolution);
extra_features=F fuses F morphology measurements into the head (MorphologyNet).
"""

from __future__ import annotations

import timm
import torch
from torch import nn


class MorphologyNet(nn.Module):
    """Backbone pooled features ++ standardised measurements -> linear classifier."""

    def __init__(self, backbone: nn.Module, extra_features: int, num_classes: int) -> None:
        super().__init__()
        self.backbone, self.norm = backbone, nn.BatchNorm1d(extra_features)  # measurements have very different scales
        self.head = nn.Linear(backbone.num_features + extra_features, num_classes)

    def forward(self, x, extra):
        return self.head(torch.cat([self.backbone(x), self.norm(extra.float())], 1))


def Net(name: str, num_classes: int = 10, pretrained: bool = True, grad_checkpointing: bool = False,
        in_chans: int = 3, extra_features: int = 0):
    model = timm.create_model(name, pretrained=pretrained, num_classes=0 if extra_features else num_classes,
                              in_chans=in_chans)
    model.set_grad_checkpointing(grad_checkpointing)  # ResNet-50 bs32 on MPS: 4.6 -> 2.5 GB, ~20% slower
    return MorphologyNet(model, extra_features, num_classes) if extra_features else model
