*[Türkçe](README.md) · **English***

# Deep Learning Based Automatic Vehicle Damage Detection for the Insurance Sector

Automatic detection, classification and localization of exterior vehicle damage
using a two-stage YOLOv11-seg architecture.

This repository holds the code, trained model weights and evaluation metrics of
a Master's thesis carried out in the Artificial Intelligence Engineering MSc
programme at the Institute of Science, Üsküdar University.

| | |
|---|---|
| **Author** | Vesalettin Burhan Özdemir (224329044) |
| **Institution** | Üsküdar University, Institute of Science |
| **Programme** | Artificial Intelligence Engineering (MSc) |
| **Year** | 2026 |

> **Note on language.** The thesis is written in Turkish, and so are the
> notebooks, the code identifiers and the runtime output of `predict.py`.
> The damage class labels are `Cizik` (scratch), `Gocuk` (dent) and
> `Kirik` (crack/break). This document and [MODELS.en.md](MODELS.en.md) are
> English translations of the Turkish originals.

---

## 1. System Architecture

The system fuses two independent instance-segmentation models:

```
                    ┌──────────────────────────┐
     Input image ───┤                          │
                    │  Model 1 — Part/Location │──► hood, door, bumper, light …
                    │  (YOLOv11m-seg)          │
                    └──────────────────────────┘
                                 │
                                 │  overlap-ratio fusion (threshold = 0.38)
                                 │
                    ┌──────────────────────────┐
     Input image ───┤                          │
                    │  Model 2 — Damage        │──► scratch, dent, crack
                    │  (YOLOv11l-seg)          │
                    └──────────────────────────┘
                                 │
                                 ▼
                  "Dent on the front left door (87% confidence)"
```

The damage mask produced by Model 2 is intersected with the part mask produced
by Model 1. If the intersection area divided by the damage area exceeds 0.38,
the damage is assigned to that part. A single damage instance may be assigned
to more than one part — for example a scratch running across both the bumper
and the fender.

---

## 2. Results

### 2.1 Final model — Model 2 (damage), held-out test set

Measured on 2,214 images / 4,458 instances.

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.7475 | 0.7544 |
| Recall | 0.6734 | 0.6563 |
| mAP@50 | 0.7422 | **0.7238** |
| mAP@50-95 | 0.5532 | 0.4743 |

Per-class mask mAP@50:

| Class | mAP@50 |
|---|---|
| Scratch (`Cizik`) | 0.6381 |
| Dent (`Gocuk`) | 0.7275 |
| Crack (`Kirik`) | 0.8058 |

Inference speed: 1.23 ms preprocessing + 6.46 ms inference + 1.15 ms
postprocessing = **8.84 ms per image** (Google Colab Pro+ GPU, 960 px).

The gap between the validation and test sets is below 0.002, which supports the
conclusion that the model has not overfitted to the validation set.

### 2.2 Model 1 (part/location), validation set

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.6568 | 0.6670 |
| Recall | 0.8249 | 0.8410 |
| mAP@50 | 0.7042 | 0.7195 |
| mAP@50-95 | 0.6145 | 0.5967 |

### 2.3 From architecture selection to the final model

| | Architecture selection (V0) | Final model (V3) |
|---|---|---|
| Model | YOLOv11-Medium | YOLOv11-Large |
| Resolution | 640 px | 960 px |
| Epochs | 60 | 150 + 70 (fp32 continuation) |
| Mask mAP@50 | 0.4325 | **0.7238** |
| Precision | 0.5290 | 0.7544 |
| Recall | 0.4513 | 0.6563 |

The full ablation table, including the intermediate versions (V1: data cleaning
and perceptual-hash deduplication, V2: overlapping tiling), is Table 7 of the
thesis. All metric files can be collected from Google Drive with
`metrik_toplayici.ipynb`.

### 2.4 Cross-validation

3-fold K-fold, mask mAP@50:

| Fold 1 | Fold 2 | Fold 3 | Mean |
|---|---|---|---|
| 0.6358 | 0.6479 | 0.6335 | **0.6391 ± 0.0078** (CV 1.22%) |

---

## 3. Repository Layout

```
.
├── README.md / README.en.md               this file
├── MODELS.md / MODELS.en.md               model card: all metrics and limitations
├── LICENSE                                MIT
├── requirements.txt                       dependencies
├── predict.py                             one-command inference script
├── .gitignore
├── metrik_toplayici.ipynb                 collects every metric file from Drive into one zip
└── notebooks/
    ├── 01_mimari_karsilastirmasi.ipynb    V0 — comparison of five YOLO variants
    ├── 02_cozunurluk_ve_kutucuklama.ipynb V1+V2 — data cleaning, 960 px, tiling
    ├── 03_nihai_model.ipynb               V3 — final model (YOLOv11l-seg, fp32) ⭐
    ├── 04_analiz_ve_degerlendirme.ipynb   test-set evaluation, real-world test
    └── 05_alan_kaymasi_denemesi.ipynb     failed expansion attempt (negative result)
```

Model weights (`.pt`) are not included in the repository because of their size;
see **[MODELS.en.md](MODELS.en.md)** for download links.

### About the notebook names

The notebooks were **copied** from the originals in Google Drive and renamed
after their actual content. The original filenames did not match the version
numbers they appeared to carry:

| Name in this repository | Original name in Drive |
|---|---|
| `01_mimari_karsilastirmasi.ipynb` | `Tez_Model.ipynb` |
| `02_cozunurluk_ve_kutucuklama.ipynb` | `Tez_Modelv3.ipynb` |
| `03_nihai_model.ipynb` | `Tez_Model_v6.ipynb` |
| `04_analiz_ve_degerlendirme.ipynb` | `Untitled2.ipynb` |
| `05_alan_kaymasi_denemesi.ipynb` | `Tez_Modelv4.ipynb` |

---

## 4. Installation

```bash
git clone https://github.com/vbrhn/tez-arac-hasar-tespit.git
cd tez-arac-hasar-tespit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the model weights into `models/` (see MODELS.en.md):

```
models/
├── model1_parca_konum_yolo11m-seg.pt
└── model2_hasar_nihai_yolo11l-seg_fp32.pt
```

---

## 5. Usage

### Single image

```bash
python predict.py --image sample.jpg
```

### Directory, with JSON output and visualizations

```bash
python predict.py \
    --image ./test_images/ \
    --out ./results/ \
    --json results.json \
    --conf 0.25
```

### Damage model only (no part assignment)

```bash
python predict.py --image sample.jpg --no-parts
```

Example output (the script prints in Turkish; `guven` = confidence,
`parca` = part, `ortusme` = overlap):

```
sample.jpg
  [1] Gocuk  guven=0.87  parca=sol_on_kapi (ortusme=0.74)
  [2] Cizik  guven=0.61  parca=sol_on_camurluk (ortusme=0.52)
```

### From Python

```python
from predict import HasarTespitSistemi      # HasarTespitSistemi = DamageDetectionSystem

sistem = HasarTespitSistemi(
    parca_model="models/model1_parca_konum_yolo11m-seg.pt",   # parca_model = part model
    hasar_model="models/model2_hasar_nihai_yolo11l-seg_fp32.pt",  # hasar_model = damage model
)
sonuc = sistem.tahmin("sample.jpg")          # tahmin() = predict()
for h in sonuc["hasarlar"]:                  # hasarlar = damages
    print(h["sinif"], h["guven"], h["parca"])   # class, confidence, part
```

Command-line flags are in English: `--image`, `--out`, `--json`, `--conf`,
`--iou`, `--imgsz`, `--overlap`, `--device`, `--no-parts`, `--draw-parts`.

---

## 6. Reproduction

The notebooks are written for Google Colab and include a Drive mount step.
`notebooks/03_nihai_model.ipynb` is sufficient to train the final model from
scratch; its cells must be run in order.

The dataset is hosted on Roboflow. The data-download cells expect a Roboflow
API key, which is **not** stored in this repository and must be supplied as an
environment variable:

```bash
export ROBOFLOW_API_KEY="..."
```

In Colab:

```python
import os
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
```

### Hardware

Training was carried out on Google Colab Pro+. Total training time for the
final model is approximately 60 hours (150 epochs fp16 + 70 epochs fp32
continuation).

---

## 7. Known Limitations

See [MODELS.en.md](MODELS.en.md) → "Limitations" for the full discussion. In
short:

1. **Cross-version comparison.** The data-splitting cell was re-run for every
   version. Although `seed=42` was fixed, the pool being split changed between
   versions, so the test sets are not identical across versions. Comparisons in
   the ablation table should therefore be read as indicative rather than
   controlled.
2. **Domain shift.** Performance drops when external data from heterogeneous
   sources is added (see `05_alan_kaymasi_denemesi.ipynb`).
3. **Scratch class.** The weakest class (0.6381); fine scratches disappear at
   low resolution.
4. **fp16 numerical instability.** Mask-loss overflow was observed under mixed
   precision; the final training run was completed in fp32 (`amp=False`).

---

## 8. Citation

```bibtex
@mastersthesis{ozdemir2026damage,
  author  = {Özdemir, Vesalettin Burhan},
  title   = {Deep Learning Based Automatic Vehicle Damage Detection
             for the Insurance Sector},
  school  = {Üsküdar University, Institute of Science},
  address = {Istanbul, Türkiye},
  year    = {2026},
  type    = {Master's Thesis},
  note    = {Original title: Sigorta Sektöründe Derin Öğrenme Tabanlı
             Otomatik Araç Hasar Tespit Sistemi},
}
```

---

## 9. License and Acknowledgements

The code is released under the MIT license. The trained model weights are for
academic use only.

This work is built on the
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library.
