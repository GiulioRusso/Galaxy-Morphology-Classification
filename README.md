# Galaxy Morphology Classification

A leakage-free, reproducible 10-class benchmark on Galaxy10 DECaLS. It compares handcrafted features, ImageNet-pretrained CNNs and a Galaxy Zoo-pretrained encoder. Every number below is generated from committed outputs.

<!-- HEADLINE:START -->
_No results yet: train with `scripts/main.py`, then run `scripts/make_results_table.py`._
<!-- HEADLINE:END -->

<img src="./images/galaxy10_example.png">

## Task and dataset

[Galaxy10 DECaLS](https://astronn.readthedocs.io/en/latest/galaxy10.html) contains 17,736 colour cutouts (256×256, g/r/z bands) from the DESI Legacy Imaging Surveys. The labels come from Galaxy Zoo DECaLS volunteer votes. All 10 classes are kept. Counts are generated from `splits/`:

<!-- DATASET:START -->
| Class | Name | Release | Used |
|---|---|---|---|
| 0 | Disturbed | 1081 | 1051 |
| 1 | Merging | 1853 | 1839 |
| 2 | Round Smooth | 2645 | 2635 |
| 3 | In-between Round Smooth | 2027 | 2021 |
| 4 | Cigar Shaped Smooth | 334 | 333 |
| 5 | Barred Spiral | 2043 | 2038 |
| 6 | Unbarred Tight Spiral | 1829 | 1819 |
| 7 | Unbarred Loose Spiral | 2628 | 2595 |
| 8 | Edge-on no Bulge | 1423 | 1419 |
| 9 | Edge-on with Bulge | 1873 | 1865 |
| | **Total** | **17736** | **17615** |

Dropped (121): 118 identical, conflicting labels, 3 blank. List: `splits/dropped.csv`.
<!-- DATASET:END -->

"Identical, conflicting labels": the release contains 59 pairs of pixel-identical images, and in every pair the two copies carry different classes. Neither label can be trusted, so both copies are dropped. "Blank": images with a single constant pixel value.

## What changed from v1

The 2023 coursework version (branch [`legacy/v1`](https://github.com/GiulioRusso/Galaxy-Morphology-Classification/tree/legacy/v1), tag `v1-coursework`) wrote rotated and flipped copies of minority-class images to disk *before* splitting. Each galaxy's 7 copies were then spread over train, validation and test. In the 10-class set, 402 of the 5,707 test images were copies of galaxies that were also in the training set. The audit is in [`scripts/audit_v1_leakage.py`](https://github.com/GiulioRusso/Galaxy-Morphology-Classification/blob/legacy/v1/scripts/audit_v1_leakage.py) and [its output](https://github.com/GiulioRusso/Galaxy-Morphology-Classification/blob/legacy/v1/scripts/audit_v1_leakage.csv). v1 also dropped the hardest class (Disturbed + Merging) from its 3-class setting.

v2 splits before any augmentation, keeps all 10 classes, removes conflicting duplicate images, and keeps overlapping cutouts of the same sky region in the same split.

## Data protocol

- **Cleaning:** blank images and pixel-identical copies with conflicting labels are found by pixel checks and logged, not hardcoded.
- **Overlap groups:** cutouts within 60″ of each other on the sky overlap and share pixels (measured correlation on the overlap ≈ 0.94–0.99). They form one group, and a group never spans two splits.
- **Split:** one stratified 70/15/15 split by galaxy group (seed 0), committed as `splits/galaxy10.csv`. Tests check disjointness and that every class appears in every split.
- **Augmentation:** applied on the fly to training images only. It uses the D4 dihedral group (90° rotations × flips; galaxies have no preferred orientation), a random 224 crop from the 256 image, and mild brightness and contrast jitter.
- **Metric:** macro-F1 over the 10 classes, plus per-class F1. Accuracy is not reported: with imbalanced classes it is dominated by the large ones.
- **Model selection:** early stopping, the best checkpoint and the model choice use validation macro-F1. Loss is logged for train and validation. The test set is read once per run.
- **Class imbalance:** weighted cross-entropy, focal loss and a class-balanced sampler are compared on B1, and the winner is picked by validation macro-F1.

## Results

One run per model (seed 0), 10 classes, test set. Validation macro-F1 is the best epoch's, which selected the checkpoint.

<!-- RESULTS:START -->
_No results yet: train with `scripts/main.py`, then run `scripts/make_results_table.py`._
<!-- RESULTS:END -->

Per-class test F1:

<!-- PERCLASS:START -->
_No results yet: train with `scripts/main.py`, then run `scripts/make_results_table.py`._
<!-- PERCLASS:END -->

Error analysis (most-confused class pairs with examples): [`notebooks/error_analysis.ipynb`](notebooks/error_analysis.ipynb).

## Comparison with published work

Published Galaxy10 DECaLS results report top-1 accuracy (about 73% to 95%), while we report macro-F1. None of them uses a duplicate-aware split, and the splits are not shared. The numbers are therefore **not comparable**. The table, protocols and caveats are in [`docs/related_work.md`](docs/related_work.md). In particular, the Zoobot encoder (B3) was pretrained on Galaxy Zoo votes that very likely include our test galaxies.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). Configs are sized for a 16 GB Apple-silicon laptop (MPS, fp16 autocast, gradient checkpointing, batch 32); CPU and CUDA also work. On a Mac, keep it plugged in with Low Power Mode off (it throttles the GPU). GPU memory is capped with `mps_memory_fraction` (default 0.4), so a run that needs too much memory fails with an out-of-memory error instead of freezing the machine.

```bash
uv sync
uv run scripts/download_data.py                        # h5 (2.7 GB, MD5-checked, resumable)
uv run scripts/preprocess.py config/preprocess.yaml    # .npy + split (the split is already committed in splits/)
uv run scripts/main.py config/b3_zoobot.yaml           # train, then test the best checkpoint
uv run scripts/make_results_table.py                   # regenerate the tables in this README
```

Training and test of a run write into one folder (layout of [AI-template](https://github.com/GiulioRusso/AI-template)). The folder suffix is the config file name, unless `name:` is set in the config:

```
experiments/<timestamp>_<config name>/
  parameters.yaml          full resolved config + git commit: reproduces the run, drives inference
  logs/
    train.log
    log_epoch.csv          one row per epoch: train loss, val loss, val macro-F1, val F1 per class, lr, seconds
    log_steps.csv          one row per training step: loss, lr
  plots/                   loss, loss_steps, macro_f1, per_class_f1, lr, val_confusion_matrix (rewritten every epoch)
  checkpoints.nosync/      best.pt (best val macro-F1), last.pt (B0: model.joblib); not committed, out of iCloud
  inference_<timestamp>/
    parameters.yaml
    logs/test.log, logs/log_test.csv   test macro-F1 and F1 per class
    predictions.csv                    image_id, label, class probabilities
    plots/                             confusion_matrix, examples (most confident correct and wrong per class)
```

Any config key can be overridden as `key.sub=value`, e.g. `data.limit=128 train.epochs=1` for a quick debug run. Re-test a run with `uv run scripts/inference.py experiments/<run>/parameters.yaml`, resume one with `train.resume=experiments/<run>/checkpoints.nosync/last.pt train.resume_optimizer=true`. Checks: `uv run pytest`, `uv run ruff check`.

| Config (`config/`) | Baseline |
|---|---|
| `b0_handcrafted.yaml` | Handcrafted geometric/texture/colour features + gradient boosting |
| `b1_resnet50_{wce,focal,sampler}.yaml` | ResNet-50 (ImageNet), one config per imbalance strategy |
| `b2_convnext_tiny.yaml` | ConvNeXt-Tiny (ImageNet-12k→1k) |
| `b3_zoobot.yaml` | Zoobot ConvNeXt-Tiny (Galaxy Zoo); same architecture as B2, different pretraining |

## Limitations

- Labels are volunteer-vote thresholds, so neighbouring classes (e.g. Disturbed vs Merging, tight vs loose spiral) overlap.
- One run per model (seed 0): differences of a few F1 points can be run-to-run noise. Cigar Shaped Smooth has only 333 images (≈50 in test), so its F1 is the noisiest.
- The data comes from a single survey (DECaLS imaging), and transfer to other surveys is not tested.
- B3's pretraining likely includes the test galaxies' Galaxy Zoo votes (see above).
