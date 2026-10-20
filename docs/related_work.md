# Published results on Galaxy10 DECaLS

**Bottom line: no published number below is directly comparable to ours.** Galaxy10 DECaLS has no canonical train/test split, and none of these papers removes duplicate galaxies before splitting. All of them report top-1 accuracy, while we report only macro-F1 and per-class F1 (accuracy is dominated by the large classes), so no number below can be read side by side with ours.

Walmsley et al. (2024, App. D.4) state the problem directly: published Galaxy10 DECaLS accuracies are hard to assess "due to dataset limitations (including duplicates, substantial class confusion, limited label precision, and lack of a canonical train/test split)".

| Work | Model | Pretraining | Top-1 acc. | Split / protocol | Comparable? |
|---|---|---|---|---|---|
| Dagli 2023, *Astroformer* (ICLR 2023, [arXiv:2304.05350](https://arxiv.org/abs/2304.05350)) | CoAtNet-style hybrid | none (from scratch) | 94.86 | Held-out test set; split fractions and validation use not stated in the paper | No: split unknown, no de-duplication |
| Same paper, Table 1 (prior work it cites) | EfficientNetV2 / DenseNet121 / EfficientNet | various | 90.24 / 88.64 / 86.00 | Protocols of the cited works vary | No |
| Walmsley et al. 2024, *Scaling Laws for Galaxy Images* ([arXiv:2404.02973](https://arxiv.org/abs/2404.02973), App. D.4) | Zoobot ConvNeXt | Galaxy Zoo (GZ Evo) + ImageNet-12k | ≈90 | Fine-tuning and linear probe; split not canonical | Closest to our B3, but split differs |
| Multimodal Universe 2024 ([arXiv:2412.02527](https://arxiv.org/abs/2412.02527), Table 3, App. B.1) | ConvNeXt-Nano | Galaxy Zoo / ImageNet-12k / none | 89.3 ± 0.1 / 83.9 ± 0.3 / 75.6 ± 1.8 | 80/20 train/test, ~15k galaxies re-rendered from Legacy Survey images (different pixels from the h5) | No: different images and split |
| Same paper | EfficientNetB0 / ResNet18 / DenseNet121 | none | 80.9 / 73.9 / 73.5 | 80/20, 10 epochs | No |
| Lagman, Naval & Reyes 2024 (SPP conf., [arXiv:2606.08826](https://arxiv.org/abs/2606.08826)) | ResNet101 / InceptionV4 | not stated | 89.4 / 88.7 | **Offline rotations/flips used to balance classes to 2,500 images each.** The paper does not say whether this happened before or after splitting; if before, it has the same leak as our v1 | No: possible leakage |

## How our protocol differs

- **Duplicates removed, overlaps grouped.** The release contains 59 pixel-identical image pairs whose two copies have different labels; we drop both copies. Cutouts within 60″ on the sky overlap and are grouped, and each group goes entirely into train, val or test (`net/data/preprocessing.py`). We use one stratified 70/15/15 split (seed 0), and the split file is committed.
- **No offline augmentation.** D4 rotations and flips are applied on the fly, to training images only.
- **Model selection on validation only.** The test set is read once per run.
- **Macro-F1 as the only metric.** The classes range from 334 to 2,645 images, and accuracy hides failures on the small ones.

## Caveat for Zoobot

Zoobot encoders are pretrained on GZ Evo, which includes Galaxy Zoo DECaLS volunteer votes ([Zoobot docs](https://zoobot.readthedocs.io/en/latest/pretrained_models.html)). Galaxy10 DECaLS labels are derived from those same GZ DECaLS votes. Our test galaxies were therefore very likely seen during pretraining, together with the vote fractions that define their labels. The Zoobot documentation does not mention excluding Galaxy10 galaxies. Zoobot is reported as **domain-pretrained, with possible label exposure**. It is not evidence of generalisation to unseen galaxies. ResNet-50 and ConvNeXt-Tiny (ImageNet only) carry no such exposure.
