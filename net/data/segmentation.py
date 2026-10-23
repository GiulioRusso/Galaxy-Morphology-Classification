"""Galaxy segmentation: each pixel -> sky, target galaxy or other source (star / neighbour).

Standard astronomical source detection (as in Source Extractor), via photutils:
1. smooth with a small Gaussian; sky level and noise sigma from sigma-clipped statistics of the cutout;
2. keep connected regions brighter than sky + `nsigma` * sigma;
3. deblend touching sources by their brightness peaks;
4. the source at the image centre is the target galaxy, every other source is "other".
The threshold follows the sky noise, not the galaxy core, so faint outskirts (arms, tidal tails) are kept.

    uv run scripts/segment.py config/segment.yaml   # -> data/segmaps.npy (uint8 N x 256 x 256) + data/morphology.npz
"""

from __future__ import annotations

import warnings

import numpy as np
from astropy.convolution import convolve
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import deblend_sources, detect_sources, make_2dgaussian_kernel

SKY, OTHER, GALAXY = 0, 1, 2
KERNEL = make_2dgaussian_kernel(2.0, size=5)


def segment(rgb: np.ndarray, nsigma: float = 1.5, npixels: int = 20, contrast: float = 0.05) -> np.ndarray:
    """uint8 HxWx3 cutout -> uint8 HxW map of SKY / OTHER / GALAXY.

    nsigma: detection threshold above the sky (lower keeps fainter outskirts, but merges more noise);
    contrast: deblending (lower splits more: too low cuts spiral arms off the galaxy, too high merges companions).
    """
    img = rgb.astype(np.float32).mean(-1)
    data = convolve(img, KERNEL)
    _, sky, sigma = sigma_clipped_stats(data, sigma=3.0)  # noise of the smoothed image, which is what is thresholded
    data -= sky
    out = np.zeros(img.shape, np.uint8)
    seg = detect_sources(data, nsigma * max(sigma, 0.1), n_pixels=npixels)  # blank images: sigma = 0
    if seg is None:  # nothing detected (e.g. blank image): all sky
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # photutils warns when a source cannot be deblended further
        seg = deblend_sources(data, seg, n_pixels=npixels, contrast=contrast, progress_bar=False)
    labels = seg.data
    cy, cx = (s // 2 for s in labels.shape)
    target = labels[cy, cx]
    if target == 0:  # centre pixel is sky: take the source closest to the centre
        yy, xx = np.nonzero(labels)
        k = np.argmin((yy - cy) ** 2 + (xx - cx) ** 2)
        target = labels[yy[k], xx[k]]
    out[labels > 0] = OTHER
    out[labels == target] = GALAXY
    return out
