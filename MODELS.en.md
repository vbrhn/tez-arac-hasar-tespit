*[Türkçe](MODELS.md) · **English***

# Model Card

Automatic vehicle damage detection system — trained model weights, training
conditions, evaluation results and limitations.

Last updated: 2026-10-13

> Class labels in the code and weights are Turkish: `Cizik` = scratch,
> `Gocuk` = dent, `Kirik` = crack/break.

---

## 1. Released Models

| File | Role | Architecture | Size |
|---|---|---|---|
| `model1_parca_konum_yolo11m-seg.pt` | **Model 1** — vehicle part segmentation | YOLOv11m-seg | 45 MB |
| `model2_hasar_nihai_yolo11l-seg_fp32.pt` | **Model 2** — damage segmentation ⭐ | YOLOv11l-seg | 56 MB |
| `model2_hasar_v1_sadece_kendi_veri_yolo11m-seg.pt` | ablation — V1 reference | YOLOv11m-seg | 45 MB |
| `model2_hasar_asama1_fp16_yolo11l-seg.pt` | intermediate checkpoint — start of the fp32 continuation | YOLOv11l-seg | 222 MB |

Only **Model 1** and **Model 2** are needed for inference. The other two exist
for reproduction and ablation verification.

`model2_hasar_asama1_fp16_yolo11l-seg.pt` is large because it still carries the
optimizer state (the continuation run was started from this file). The final
model is 56 MB after `strip_optimizer`.

### Download

The weights are not kept in the repository because of GitHub file-size limits.

| Source | Link |
|---|---|
| Google Drive | `<share link here>` |
| Zenodo (DOI) | `<DOI here>` |

---

## 2. Model 2 — Damage Segmentation (final)

### 2.1 Purpose

Extracts damage regions from an image of a vehicle exterior at pixel level and
assigns them to one of three classes.

| Class id | Label | Description |
|---|---|---|
| 0 | `Cizik` — scratch | Linear mark on the paint surface |
| 1 | `Gocuk` — dent | Deformation of the sheet-metal surface |
| 2 | `Kirik` — crack | Breakage, cracking, missing part |

### 2.2 Training data

| | Value |
|---|---|
| Training images | 15,490 |
| Split ratio | 70% / 15% / 15% |
| Test images | 2,214 (4,458 instances) |
| Share of own (original) data | 79% |
| External source | Roboflow `cocotoyolo-cmnax/damage_2-w8sdi` (dent class only) |

Data preparation steps:

- **Perceptual-hash deduplication** (dHash, Hamming distance ≤ 3)
- **Overlapping tiling** — 2×2 grid, 20% overlap, polygon clipping with
  Shapely; 5,163 → 24,568 instances (4.8× increase)
- **Oversampling** — minority classes duplicated to disk with an `OS` prefix

The `scratch` and `crack` definitions in the external dataset conflicted with
our own `Cizik` / `Kirik` labels, so those two classes were **not** taken from
the external source; only the `dent` → `Gocuk` mapping was used.

### 2.3 Training conditions

Training was done in two stages. The first stage was planned for 150 epochs in
fp16; a NaN overflow appeared in the mask loss at epoch 83, and training was
resumed in fp32 from the `best.pt` of epoch 82.

| Parameter | Stage 1 (fp16) | Stage 2 (fp32) |
|---|---|---|
| Starting weights | `yolo11l-seg.pt` (COCO) | stage-1 `best.pt` (epoch 82) |
| Epochs | 150 (stopped at 83) | 70 |
| `amp` | `True` | **`False`** |
| `lr0` | 0.001 | 0.0004 |
| `lrf` | 0.01 | 0.01 |
| Optimizer | AdamW | AdamW |
| `cos_lr` | `True` | `True` |
| `warmup_epochs` | 3.0 | 2.0 |
| Image size | 960 | 960 |
| Batch size | 6 | 6 |
| `patience` | 35 | 35 |
| `close_mosaic` | 20 | 20 |

Augmentation (identical in both stages):

```
hsv_h=0.015   hsv_s=0.7     hsv_v=0.5
fliplr=0.5    flipud=0.0    degrees=5.0
mosaic=0.5    copy_paste=0.3
scale=0.4     translate=0.1 erasing=0.2
```

Loss weights: `box=8.0`, `cls=0.5`, `dfl=1.5`

**Why fp32.** The lower bound of fp16 is about 10⁻⁸, that of fp32 about 10⁻³⁸.
When very small intermediate values arise in the mask-loss computation, fp16
underflows and produces NaN. `amp=False` removes that risk at the cost of
roughly 1.6× slower training.

The learning rate for the continuation was set to 0.0004 rather than 0.001 to
avoid destroying the already-learned representations with large gradient steps
(catastrophic forgetting).

**Note.** Mask mAP@50 measured zero in the first epoch of the continuation: the
mask head in the stage-1 `best.pt` had been affected by the NaN overflow. The
mask head relearned from scratch and surpassed the previous best at epoch 33.

| | |
|---|---|
| Total training time | ~60 hours (Google Colab Pro+) |
| Parameters | 27,586,905 |

### 2.4 Evaluation — held-out test set

2,214 images / 4,458 instances.

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.7475 | 0.7544 |
| Recall | 0.6734 | 0.6563 |
| mAP@50 | 0.7422 | **0.7238** |
| mAP@50-95 | 0.5532 | 0.4743 |

Per-class mask mAP@50:

| Class | mAP@50 |
|---|---|
| Scratch | 0.6381 |
| Dent | 0.7275 |
| Crack | 0.8058 |

Confusion-matrix diagonal (confidence 0.25, IoU 0.45): scratch 0.637,
dent 0.737, crack 0.785. Background wrongly counted as damage:
403 + 622 + 311 = 1,336 cases.

Inference speed (960 px): 1.23 ms preprocessing + 6.46 ms inference + 1.15 ms
postprocessing = **8.84 ms per image**.

### 2.5 Cross-validation

3-fold K-fold, mask mAP@50:

| Fold | mAP@50 |
|---|---|
| 1 | 0.6358 |
| 2 | 0.6479 |
| 3 | 0.6335 |
| **Mean** | **0.6391 ± 0.0078** |

Coefficient of variation 1.22% — consistency across folds is high. The K-fold
figures are lower than the final model because cross-validation was run on a
smaller data pool and for fewer epochs.

---

## 3. Model 1 — Part/Location Segmentation

| | |
|---|---|
| Architecture | YOLOv11m-seg |
| Dataset | Carparts-Seg |
| Image size | 960 px |
| Role | Determining which vehicle part the damage lies on |

Validation set:

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.6568 | 0.6670 |
| Recall | 0.8249 | 0.8410 |
| mAP@50 | 0.7042 | 0.7195 |
| mAP@50-95 | 0.6145 | 0.5967 |

This model is independent of damage detection and was not retrained. Recall
being markedly higher than precision is the desired behaviour for the fusion
step: missing a part is more costly than proposing an extra one.

---

## 4. Fusion

With `H` the damage mask and `P` the part mask, the overlap ratio is:

```
overlap(H, P) = area(H ∩ P) / area(H)
```

Every part with `overlap ≥ 0.38` is assigned to that damage. The threshold was
chosen on the validation set to balance part-assignment accuracy against the
number of spurious assignments.

A damage instance may be assigned to several parts. If no part clears the
threshold, the damage is reported with `parca=None`.

---

## 5. Ablation

| Version | Change | Mask mAP@50 |
|---|---|---|
| V0 | Architecture selection — YOLOv11m-seg, 640 px, 60 epochs | 0.4325 |
| V1 | Data cleaning, deduplication | Table 7 of the thesis |
| V2 | 960 px + overlapping tiling | Table 7 of the thesis |
| V3 | YOLOv11l-seg, fp32 continuation | **0.7238** |

For V0, precision is 0.5290 and recall 0.4513. Per-class metrics were not
recorded in the V0 run (the `sinif_mAP50` field was not written at that time),
so no class breakdown is available.

The numeric values for the intermediate versions are given in Table 7 of the
thesis; `metrik_toplayici.ipynb` collects every metric JSON file from Drive
into a single zip.

---

## 6. Limitations

### 6.1 Validity of cross-version comparison

The data-splitting cell was re-run for every version. Although `seed=42` was
fixed, the contents of the pool being split changed between versions, so the
resulting train/validation/test sets are not identical. This introduces a
two-way bias: an image in one version's test set may appear in another
version's training set, and the class distributions of the splits differ.

Consequence: the version comparisons in the ablation table are **indicative**,
not a controlled comparison on a fixed test set. The final model's (V3) figures
on its own test set remain valid.

This could not be corrected retroactively; retraining every version on a single
fixed split would require more than 60 hours of additional compute.

### 6.2 Domain shift

When data from three different external sources was added, performance fell
from 0.6894 to 0.3319. The causes are differing annotation conventions across
sources, conflicting class definitions, and differences in image-acquisition
conditions. The final model uses only one external source and only one class
from it (`dent` → `Gocuk`). The experiment is in
`05_alan_kaymasi_denemesi.ipynb`.

### 6.3 Per-class performance gap

The scratch class performs worst (0.6381). Fine scratches disappear at low
resolution and under poor lighting. The crack class performs best (0.8058);
cracks are the most visually distinct damage type.

### 6.4 Background false positives

Background was marked as damage in 1,336 test-set cases. Reflections, shadows
and dirt are the main sources. In an insurance setting this means a human
adjuster must verify the output; the system is decision support, not an
autonomous decision maker.

### 6.5 Out of scope

- Interior damage (upholstery, dashboard) is out of scope.
- Mechanical and structural damage cannot be detected from images.
- Damage severity (repair cost) is not estimated; only damage type and location
  are produced.
- The model was trained on the Turkish vehicle fleet and on Turkish adjuster
  photography conventions; performance in other geographies has not been
  verified.

### 6.6 Compute

Inference speed was measured on a Colab GPU. Real-time operation on CPU or on
mobile devices has not been verified.

---

## 7. Ethics and Intended Use

The system is designed as **decision support** within the insurance adjustment
process. It is not intended to decide claims automatically; its outputs must be
verified by a qualified adjuster.

The training images show vehicle exteriors; frames containing license plates or
identifiable people were removed during data preparation.

---

## 8. Reproduction

| Step | File |
|---|---|
| Architecture selection (V0) | `notebooks/01_mimari_karsilastirmasi.ipynb` |
| Data cleaning + tiling (V1, V2) | `notebooks/02_cozunurluk_ve_kutucuklama.ipynb` |
| Final model (V3) | `notebooks/03_nihai_model.ipynb` |
| Test evaluation | `notebooks/04_analiz_ve_degerlendirme.ipynb` |
| Domain-shift experiment | `notebooks/05_alan_kaymasi_denemesi.ipynb` |
| Metric collection | `metrik_toplayici.ipynb` |

The dataset lives on Roboflow; a `ROBOFLOW_API_KEY` environment variable is
required for access.
