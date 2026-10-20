import numpy as np
import torch

from net.data.transforms.transforms import build_transforms, d4
from net.utils.config import ROOT, load_config


def test_d4_is_a_group_element():
    x = torch.arange(2 * 5 * 5, dtype=torch.float32).reshape(2, 5, 5)
    group = [torch.rot90(x, k, (-2, -1)) for k in range(4)]
    group += [g.flip(-1) for g in group]
    torch.manual_seed(0)
    for _ in range(50):
        y, label = d4(x, 3)
        assert y.shape == x.shape and label == 3
        assert any(torch.equal(y, g) for g in group)


def test_config_transforms_shapes():
    cfg = load_config([str(ROOT / "config/resnet50_wce.yaml")])
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    for stage in ("train", "eval"):
        out, label = build_transforms(cfg["transforms"][stage])(img, 7)
        assert out.shape == (3, 224, 224) and out.dtype == torch.float32 and label == 7
