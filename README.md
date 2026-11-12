# BraTS Tumor Growth Projection

Extension of a U-Net brain tumor segmentation project: given a tumor-core
segmentation for a BraTS 2021 case, measure its volume and project how it
could grow over time using published population growth models — then
evaluate how segmentation error propagates into the growth projection.

## Files

- `tumor_growth_brats.py` — Core module. Adapts the LGG U-Net to BraTS 2021:
  3 input channels (T1ce, T2, FLAIR), tumor core = labels 1 + 4, slices
  cached once for training, validation split by case (patient), whole-case
  3D inference reporting tumor-core volume in mL and 3D Dice vs the expert
  segmentation. Provides `load_case`, `predict_case`, `load_trained_model`,
  and the shared paths/constants the other two files import.
  Improvements over the LGG version: detection threshold 0.5 (matching the
  validation metric), brain mask taken from the skull-stripped scan itself
  instead of a fixed ellipse, no Gaussian blur on the probability map, and
  the best saved weights are reloaded for inference.
- `val_cases.txt` — The 15 held-out validation case IDs from the current
  training run (split by patient, seed 42).
- `growth_projection.py` — Projects tumor-core volume V(t) with three
  models from Stensjoen et al. (2015) and draws the projected volume by
  expanding the current mask evenly (in mm, via distance transform)
  inside the brain.
- `evaluate_cases.py` — For every held-out case, compares U-Net vs expert
  tumor-core volume, computes 3D Dice, checks size-group agreement, and
  reports how the volume difference carries into the projected volume.
  Writes `results/eval_volumes.csv`.

## Growth models (Stensjoen et al. 2015, Neuro-Oncology 17(10):1402–1411)

- Linear radial growth (Table 2): radius grows ~0.008314 cm/day
- Gompertz (Table 2): K = 158.04 mL, alpha = 0.007545 /day
- Size-group median specific growth rate: 3.9 / 1.2 / 0.3 %/day for
  tumors < 3.88 / 3.88–36.88 / ≥ 36.88 mL

Tumor core is approximated as BraTS labels 1 + 4.

## Limits — read before interpreting any figure

- These are **population** fits for **untreated glioblastoma**. Individual
  tumors in the paper varied widely (SGR −1.9 to 13.2 %/day); about a
  third did not grow.
- The paper's scan intervals were 14–98 days. Projections beyond 98 days
  are extrapolation (the script warns when you do this).
- Models predict volume only. The even outward expansion is a display
  choice, not a prediction of growth shape or direction.

## Usage

```
pip install -r requirements.txt
python tumor_growth_brats.py --retrain     # train (writes val_cases.txt)
python tumor_growth_brats.py               # run on held-out cases
python growth_projection.py --case BraTS2021_00000 --days 30
python growth_projection.py --days 30 --source unet
python evaluate_cases.py            # all held-out cases, 30-day projection
python evaluate_cases.py --days 60
```

## Dataset

BraTS 2021 — **not** included in this repository and must not be
redistributed. Download it from the official BraTS source under its terms
and place it in this layout (the code looks for it here):

```
data/
  BraTS2021_00000/
    BraTS2021_00000_t1ce.nii.gz
    BraTS2021_00000_t2.nii.gz
    BraTS2021_00000_flair.nii.gz
    BraTS2021_00000_seg.nii.gz   # expert segmentation, if available
```

## Results

Held-out evaluation from `evaluate_cases.py` (15 cases in `val_cases.txt`,
U-Net vs expert tumor-core segmentation; growth projection at 30 days).
Model: trained 30 epochs on 85 cases (100 cases total, split by patient,
seed 42); best checkpoint at epoch 23 (validation slice Dice 0.7098).

- **Mean 3D Dice: 0.815** (median 0.896)
- Mean volume difference (bias): **+1.4 mL**
- Mean absolute volume difference: 3.9 mL (median 9%)
- Same Stensjøen size group as expert: **12 / 15 cases**

| Case | Expert (mL) | U-Net (mL) | Diff % | 3D Dice | Size group (expert / U-Net) |
|---|---|---|---|---|---|
| BraTS2021_00002 | 34.9 | 28.6 | −18% | 0.877 | medium / medium |
| BraTS2021_00014 | 2.6 | 1.0 | −63% | 0.481 | small / small |
| BraTS2021_00016 | 23.3 | 28.2 | +21% | 0.895 | medium / medium |
| BraTS2021_00021 | 7.6 | 3.3 | −57% | 0.515 | medium / small † |
| BraTS2021_00061 | 9.3 | 9.5 | +2% | 0.911 | medium / medium |
| BraTS2021_00062 | 6.3 | 6.2 | −1% | 0.968 | medium / medium |
| BraTS2021_00074 | 45.0 | 46.9 | +4% | 0.951 | large / large |
| BraTS2021_00085 | 1.9 | 1.4 | −25% | 0.727 | small / small |
| BraTS2021_00087 | 20.6 | 15.7 | −24% | 0.831 | medium / medium |
| BraTS2021_00099 | 18.6 | 18.6 | 0% | 0.944 | medium / medium |
| BraTS2021_00104 | 35.8 | 39.1 | +9% | 0.929 | medium / large † |
| BraTS2021_00106 | 71.1 | 71.9 | +1% | 0.901 | large / large |
| BraTS2021_00107 | 20.9 | 19.9 | −5% | 0.896 | medium / medium |
| BraTS2021_00112 | 16.8 | 44.6 | +165% | 0.475 | medium / large † |
| BraTS2021_00128 | 45.0 | 46.2 | +3% | 0.931 | large / large |

† Size group differs, so a size-group growth projection for this case
would use a different growth rate than the expert volume would.

Notes on reading these numbers:

- The mean Dice (0.815) is pulled down by the same three outlier cases
  as in the earlier shorter training run — 00021 (0.515), 00014 (0.481)
  and 00112 (0.475). The median case scores 0.896, and 10 of 15 cases
  score ≥ 0.87. Retraining to the full 30-epoch schedule improved Dice
  on 14 of 15 cases (one unchanged) versus the earlier run
  (mean 0.775, median 0.888).
- The weakest cases are still the smallest tumors (00014: 2.6 mL,
  00085: 1.9 mL), where a few voxels of error cost a lot of Dice — plus
  one persistent large over-segmentation (00112, predicted 44.6 mL vs
  16.8 mL expert). Longer training improved the small-tumor cases
  (00021: 0.392 → 0.515; 00085: 0.609 → 0.727) but did not fix them,
  which points to architecture/threshold limits rather than
  training length.
- Overall volume bias is small (+1.4 mL), but that averages out large
  per-case errors in both directions (dominated by 00112, +27.8 mL);
  the per-case table above is the honest picture.
- Because 12/15 cases land in the correct size group, the growth
  projection usually starts from the right growth-rate group; the three
  mismatched cases (marked †) are where segmentation error most changes
  the projected volume.

## ⚠ Research / educational use only — not for clinical use.
