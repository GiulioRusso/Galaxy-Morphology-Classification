"""Model: any timm backbone with a fresh head. Arguments are the keys under `model:` in the config.

Zoobot encoders load the same way: name="hf_hub:mwalmsley/zoobot-encoder-convnext_tiny".
"""

from __future__ import annotations

import timm


def Net(name: str, num_classes: int = 10, pretrained: bool = True, grad_checkpointing: bool = False):
    model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    model.set_grad_checkpointing(grad_checkpointing)  # ResNet-50 bs32 on MPS: 4.6 -> 2.5 GB, ~20% slower
    return model
