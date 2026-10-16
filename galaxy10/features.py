"""B0: handcrafted features + gradient boosting.

Port of v1 AIA/handcrafted_features_extraction.py (pipeline: NLM denoise -> Otsu -> opening -> central
object -> geometric / intensity / GLCM / LBP / Gabor / Fourier features), with v1 bugs fixed:
- ellipse axes used the wrong contour (moments[i]) and a wrong eigenvalue formula;
- LBP 'uniform' codes take P+2=10 values but were binned into 256 bins; blurred images were computed, unused;
- Gabor wavelengths were < 1 px; kernels were divided by their (near-zero) sum;
- "Fourier descriptors" were sampled contour points, not Fourier coefficients;
- colour was discarded; now per-channel statistics inside the galaxy mask are included.
"""
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from galaxy10.evaluate import PROB_COLS, evaluate_run

GLCM_ANGLES = np.arange(4) * np.pi / 4
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
GABOR = [(t, lam) for t in np.arange(4) * np.pi / 4 for lam in (4, 8, 16)]
N_FOURIER = 10


def central_mask(gray):
    """Binary mask of the Otsu object closest to the image centre (whole image if none)."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    n, lab, _, cent = cv2.connectedComponentsWithStats(mask)
    if n == 1:
        return np.full_like(mask, 255)
    c = np.array(gray.shape[::-1]) / 2
    k = 1 + np.argmin(((cent[1:] - c) ** 2).sum(1))
    return np.where(lab == k, 255, 0).astype(np.uint8)


def extract(rgb):
    gray = cv2.fastNlMeansDenoising(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None, 10, 7, 21)
    mask = central_mask(gray)
    contour = max(cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    m = cv2.moments(mask, binaryImage=True)
    area, perim = m["m00"], cv2.arcLength(contour, True)
    hu = cv2.HuMoments(m).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    # ellipse axes from the eigenvalues of the second central moments
    half_tr, diff = (m["mu20"] + m["mu02"]) / 2, np.hypot((m["mu20"] - m["mu02"]) / 2, m["mu11"])
    axis_ratio = np.sqrt(max(half_tr - diff, 0) / (half_tr + diff + 1e-12))
    (_, (w, h), _) = cv2.minAreaRect(contour)
    x, y, bw, bh = cv2.boundingRect(contour)
    hull = cv2.contourArea(cv2.convexHull(contour))
    geometric = [area, perim, 4 * np.pi * area / (perim ** 2 + 1e-12), axis_ratio,
                 min(w, h) / (max(w, h) + 1e-12), area / (bw * bh + 1e-12), area / (hull + 1e-12), *hu]

    inside = gray[mask > 0].astype(float)
    hist = np.bincount(inside.astype(int), minlength=256) / len(inside)
    nz = hist[hist > 0]
    colour = rgb[mask > 0].astype(float)
    intensity = [inside.mean(), inside.std(), (hist ** 2).sum(), -(nz * np.log2(nz)).sum(),
                 *colour.mean(0), *colour.std(0), *(colour.mean(0) / (colour.mean() + 1e-12))]

    roi = gray[y:y + bh, x:x + bw]
    glcm = graycomatrix(roi // 8, [1], GLCM_ANGLES, levels=32, normed=True)
    texture = [graycoprops(glcm, p).ravel() for p in GLCM_PROPS]
    for r in (1, 3, 5):
        lbp = local_binary_pattern(roi, 8, r, "uniform")
        texture.append(np.bincount(lbp.astype(int).ravel(), minlength=10) / lbp.size)
    roi_f = roi.astype(np.float32)
    for theta, lam in GABOR:
        k = cv2.getGaborKernel((31, 31), 0.56 * lam, theta, lam, 1.0, 0)
        resp = np.abs(cv2.filter2D(roi_f, cv2.CV_32F, k - k.mean()))
        texture.append([resp.mean(), resp.std()])

    z = contour[:, 0, 0] + 1j * contour[:, 0, 1]
    coef = np.abs(np.fft.fft(z - z.mean()))
    fourier = np.zeros(N_FOURIER)
    n = min(N_FOURIER, len(coef) - 2)
    if n > 0:
        fourier[:n] = coef[2:2 + n] / (coef[1] + 1e-12)  # translation/rotation/scale invariant

    return np.concatenate([geometric, intensity, np.concatenate([np.ravel(t) for t in texture]), fourier])


_images = None


def _init(images_path):
    global _images
    _images = np.load(images_path, mmap_mode="r")


def _extract_id(i):
    return extract(_images[i])


def compute(images_path, ids, workers=None):
    with Pool(workers, initializer=_init, initargs=(images_path,)) as pool:
        return np.stack(pool.map(_extract_id, ids, chunksize=64))


def run(cfg, seed, run_dir):
    df = pd.read_csv(Path(cfg["splits"]) / f"galaxy10_seed{seed}.csv")
    if cfg.get("limit"):
        df = df.groupby("split", group_keys=False).apply(lambda s: s.sample(min(cfg["limit"], len(s)), random_state=0))
    cache = Path(cfg["images"]).parent / "features.npz"
    if cache.exists() and np.isin(df.image_id, np.load(cache)["ids"]).all():
        z = np.load(cache)
        X = pd.DataFrame(z["X"], index=z["ids"]).loc[df.image_id].to_numpy()
    else:
        X = compute(cfg["images"], df.image_id.to_numpy())
        if not cfg.get("limit"):
            np.savez(cache, ids=df.image_id.to_numpy(), X=X)
    tr, va, te = (df.split == s for s in ("train", "val", "test"))
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, class_weight="balanced",
                                         early_stopping=True, random_state=seed)  # internal val is taken from train
    clf.fit(X[tr], df.label[tr])
    val_f1 = f1_score(df.label[va], clf.predict(X[va]), average="macro")
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([dict(epoch=clf.n_iter_, val_macro_f1=val_f1)]).to_csv(run_dir / "train_log.csv", index=False)
    probs = np.zeros((te.sum(), len(PROB_COLS)))
    probs[:, clf.classes_] = clf.predict_proba(X[te])  # classes absent from a tiny debug train set stay 0
    out = pd.DataFrame(probs, columns=PROB_COLS)
    out.insert(0, "label", df.label[te].to_numpy())
    out.insert(0, "image_id", df.image_id[te].to_numpy())
    out.to_csv(run_dir / "test_probs.csv", index=False, float_format="%.5f")
    return evaluate_run(run_dir)
