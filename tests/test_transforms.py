import numpy as np
import torch

from galaxy10.transforms import D4, build


def test_d4_is_a_group_element():
    x = torch.arange(2 * 5 * 5, dtype=torch.float32).reshape(2, 5, 5)
    group = [torch.rot90(x, k, (-2, -1)) for k in range(4)]
    group += [g.flip(-1) for g in group]
    torch.manual_seed(0)
    for _ in range(50):
        y = D4()(x)
        assert y.shape == x.shape
        assert any(torch.equal(y, g) for g in group)


def test_build_shapes():
    img = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    for train in (True, False):
        out = build(train)(img)
        assert out.shape == (3, 224, 224) and out.dtype == torch.float32
