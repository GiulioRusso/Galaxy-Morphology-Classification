"""Morphology measurements of the target galaxy (inside its segmentation region, see segmentation.py).

- statmorph (Rodriguez-Gomez et al. 2019): concentration C, asymmetry A, smoothness S, Gini, M20, Gini-M20
  merger and bulge statistics, multimode, intensity, deviation, outer asymmetry, shape asymmetry, ellipticity,
  elongation, Sersic index, r20/r80 in units of the Petrosian radius;
- azimuthal Fourier modes: A_m = |sum I exp(-i m theta)| / sum I for m = 1..4 in three radial annuli.
  m=1 lopsidedness (disturbed, merging), m=2 bars and two-armed spirals, m=3,4 multi-armed / flocculent structure.
  Amplitudes only, so every value is rotation invariant.

Galaxy10 images are 8-bit RGB, not calibrated flux: the values are relative, fine as classifier inputs, not as
physical measurements. Failed or infinite values are filled with the train-split median in prepare().
"""

from __future__ import annotations

import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import statmorph

from net.data.segmentation import GALAXY, OTHER, segment
from net.utils.logger import logger

STATMORPH = ["concentration", "asymmetry", "smoothness", "gini", "m20", "gini_m20_merger", "gini_m20_bulge",
             "multimode", "intensity", "deviation", "outer_asymmetry", "shape_asymmetry", "ellipticity_asymmetry",
             "elongation_asymmetry", "sersic_n"]
MODES, ANNULI = (1, 2, 3, 4), ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0))  # annuli in units of the half-light radius
NAMES = STATMORPH + ["r20_petro", "r80_petro", "flag"] + [f"A{m}_r{i}" for i in range(len(ANNULI)) for m in MODES]


def fourier_modes(img: np.ndarray, mask: np.ndarray, xc: float, yc: float, r_half: float) -> list[float]:
    yy, xx = np.indices(img.shape)
    r, theta = np.hypot(xx - xc, yy - yc) / max(r_half, 1.0), np.arctan2(yy - yc, xx - xc)
    out = []
    for lo, hi in ANNULI:
        sel = mask & (r >= lo) & (r < hi)
        flux, ang = img[sel], theta[sel]
        total = flux.sum()
        out += [abs((flux * np.exp(-1j * m * ang)).sum()) / total if total > 0 else np.nan for m in MODES]
    return out


def measure(rgb: np.ndarray, segmap: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 cutout + its segmentation map -> float32 vector, one value per name in NAMES."""
    img = rgb.astype(np.float64).mean(-1)
    img -= np.median(img[segmap == 0]) if (segmap == 0).any() else 0.0  # subtract the sky level
    galaxy = segmap == GALAXY
    if galaxy.sum() < 20:
        return np.full(len(NAMES), np.nan, np.float32)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = statmorph.source_morphology(img, galaxy.astype(int), gain=1.0, mask=segmap == OTHER)[0]
        values = [getattr(m, k) for k in STATMORPH]
        values += [m.r20 / m.rpetro_circ, m.r80 / m.rpetro_circ, m.flag]
        values += fourier_modes(np.clip(img, 0, None), galaxy, m.xc_centroid, m.yc_centroid, m.r50)
    except Exception:  # statmorph raises on degenerate sources
        return np.full(len(NAMES), np.nan, np.float32)
    return np.asarray(values, np.float32)


_images, _seg_kwargs = None, None


def _init(images_path, seg_kwargs):
    from threadpoolctl import threadpool_limits

    threadpool_limits(1)  # one process per core already: nested BLAS threads would oversubscribe the CPU
    global _images, _seg_kwargs
    _images, _seg_kwargs = np.load(images_path, mmap_mode="r"), seg_kwargs


def _one(i):
    rgb = np.array(_images[i])
    segmap = segment(rgb, **_seg_kwargs)
    return segmap, measure(rgb, segmap)


def prepare(images_path, split_csv, segmaps_out, morphology_out, seg_kwargs: dict, workers=None) -> None:
    """Segment every image of the split CSV and measure it, once (~20 min on 8 processes).

    segmaps_out: uint8 .npy indexed by image_id like the images (0 sky, 1 other source, 2 galaxy; ids outside the
    split stay 0). morphology_out: .npz with ids, X (one row per id, columns NAMES), names. Failed measurements are
    filled with the train-split median and flagged with flag = 4.
    """
    rows = pd.read_csv(split_csv)
    shape = np.load(images_path, mmap_mode="r").shape[:3]
    segmaps = np.lib.format.open_memmap(segmaps_out, mode="w+", dtype=np.uint8, shape=shape)
    X = np.empty((len(rows), len(NAMES)), np.float32)
    with Pool(workers, initializer=_init, initargs=(images_path, seg_kwargs)) as pool:
        for k, (segmap, x) in enumerate(pool.imap(_one, rows.image_id, chunksize=32)):
            segmaps[rows.image_id.iloc[k]], X[k] = segmap, x
            if k % 2000 == 0:
                logger.info(f"segmented and measured {k}/{len(rows)}")
    segmaps.flush()
    X[np.isinf(X)] = np.nan
    failed = np.isnan(X).all(1)
    X[failed, NAMES.index("flag")] = 4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # all-NaN column (tiny test sets): median is NaN, then 0 below
        X = np.where(np.isnan(X), np.nanmedian(X[(rows.split == "train").to_numpy()], 0), X)
    X = np.nan_to_num(X)
    np.savez(morphology_out, ids=rows.image_id.to_numpy(), X=X, names=np.array(NAMES))
    logger.info(f"{len(rows)} images, {failed.sum()} measurements failed -> {segmaps_out}, {morphology_out}")


def load(path, ids) -> np.ndarray:
    """Rows of the morphology cache for image ids, in order."""
    z = np.load(path)
    return pd.DataFrame(z["X"], index=z["ids"]).loc[ids].to_numpy(np.float32)
