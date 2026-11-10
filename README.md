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

[TODO: paste the summary from `evaluate_cases.py` here — mean 3D Dice,
mean volume bias, and size-group agreement across held-out cases.]

## ⚠ Research / educational use only — not for clinical use.
