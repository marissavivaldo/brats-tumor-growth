"""
growth_projection.py
--------------------
Add-on to tumor_growth_brats.py (that file is not modified; this one imports it).
Adds a third panel: projected tumor-core growth after a chosen number of days,
using the population growth models fitted by:

  Stensjoen AL et al. (2015). Growth dynamics of untreated glioblastomas in vivo.
  Neuro-Oncology 17(10):1402-1411. doi:10.1093/neuonc/nov029

What it does
------------
1. Gets the tumor-core mask for one BraTS case: the expert segmentation
   (default) or the U-Net's prediction (--source unet).
2. Measures its volume V0 in mL from the 3D mask and the voxel size.
3. Projects V(t) with three models from the paper:
     Linear radial (Table 2):   V(t) = (4pi/3) * (r0 + a*t)^3,      a = 0.008314 cm/day
     Gompertz (Table 2):        V(t) = K * exp[ln(V0/K) * exp(-a*t)], a = 0.007545 /day, K = 158.04 mL
     Size-group median SGR      V(t) = V0 * exp(SGR * t), SGR = 3.9 / 1.2 / 0.3 %/day
       (Results, Fig. 2):       for V0 < 3.88 / 3.88-36.88 / >= 36.88 mL
4. Draws each projected volume by expanding the current tumor outward evenly
   (in mm) inside the brain until it reaches that volume.

Limits (also shown on the figure)
---------------------------------
- These are POPULATION fits for UNTREATED glioblastoma. Individual tumors in
  the paper varied widely (SGR -1.9 to 13.2 %/day); about a third did not grow.
- The paper's scan intervals were 14-98 days. Beyond 98 days is extrapolation.
- The models predict volume only. The even outward expansion is a display
  choice, not a prediction of shape or direction.
- The paper measured enhancing + central non-enhancing tumor on contrast T1.
  Here that is approximated by BraTS labels 1 + 4 (tumor core).

Usage
-----
    python growth_projection.py --case BraTS2021_00000 --days 30
    python growth_projection.py --days 60                 # first held-out case
    python growth_projection.py --days 30 --source unet   # use the U-Net's mask
"""

import os
import sys
import argparse
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from skimage import measure

import tumor_growth_brats as tg

# ──────────────────────────────────────────────
# Parameters from Stensjoen et al. 2015
# ──────────────────────────────────────────────
MAX_OBSERVED_DAYS        = 98        # longest scan interval in the paper
RADIAL_ALPHA_CM_PER_DAY  = 0.008314  # Table 2 (~30 mm/year, matches median VRE 29.6 mm/yr)
GOMPERTZ_ALPHA_PER_DAY   = 0.007545  # Table 2
GOMPERTZ_K_ML            = 158.04    # Table 2
SIZE_GROUPS = [                      # (upper volume bound in mL, median SGR in %/day)
    (3.88,   3.9),
    (36.88,  1.2),
    (np.inf, 0.3),
]
REPORT_DAYS = (14, 30, 60, 98)


def v_radial(v0, t):
    r0 = (3 * v0 / (4 * np.pi)) ** (1 / 3)          # cm (1 mL = 1 cm^3)
    return 4 * np.pi / 3 * (r0 + RADIAL_ALPHA_CM_PER_DAY * t) ** 3


def v_gompertz(v0, t):
    return GOMPERTZ_K_ML * np.exp(np.log(v0 / GOMPERTZ_K_ML) * np.exp(-GOMPERTZ_ALPHA_PER_DAY * t))


def size_group_sgr(v0):
    for upper, sgr in SIZE_GROUPS:
        if v0 < upper:
            return sgr


def v_size_group(v0, t):
    return v0 * np.exp(size_group_sgr(v0) / 100 * t)


MODELS = [
    ("Linear radial",          v_radial,     "#58a6ff"),
    ("Gompertz",               v_gompertz,   "#3fb950"),
    ("Size-group median SGR",  v_size_group, "#d29922"),
]


# ──────────────────────────────────────────────
# Drawing a projected volume
# ──────────────────────────────────────────────
def grow_mask(mask, brain, zooms, target_ml):
    """Adds the brain voxels closest (in mm) to the current tumor until the
    mask reaches target_ml. Returns (grown_mask, hit_brain_limit)."""
    voxel_ml = float(np.prod(zooms)) / 1000.0
    n_now    = int(mask.sum())
    n_target = int(round(target_ml / voxel_ml))
    if n_target <= n_now:
        return mask.copy(), False

    dist = distance_transform_edt(~mask, sampling=zooms)
    cand = np.flatnonzero(brain & ~mask)
    n_add = min(n_target - n_now, cand.size)
    nearest = np.argpartition(dist.ravel()[cand], n_add - 1)[:n_add]

    grown = mask.copy().ravel()
    grown[cand[nearest]] = True
    return grown.reshape(mask.shape), n_add < (n_target - n_now)


def draw_contours(ax, mask2d, color, lw, ls="-"):
    for c in measure.find_contours(mask2d.astype(float), 0.5):
        ax.plot(c[:, 1], c[:, 0], color=color, linewidth=lw, linestyle=ls)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",   help="BraTS case, e.g. BraTS2021_00000 "
                                         "(default: first case in val_cases.txt)")
    parser.add_argument("--days",   type=float, default=30)
    parser.add_argument("--source", choices=["expert", "unet"], default="expert",
                        help="Which tumor mask to project from")
    args = parser.parse_args()

    case_id = args.case
    if case_id is None:
        if not os.path.exists(tg.VAL_LIST_PATH):
            print("ERROR: give --case, or run tumor_growth_brats.py --retrain first.")
            sys.exit(1)
        with open(tg.VAL_LIST_PATH) as f:
            case_id = f.readline().strip()

    t = args.days
    extrapolated = t > MAX_OBSERVED_DAYS
    print(f"\n  Case: {case_id}   Days: {t:g}   Mask source: {args.source}")
    if extrapolated:
        print(f"  ⚠ {t:g} days is beyond the paper's longest interval "
              f"({MAX_OBSERVED_DAYS} days): extrapolation.")

    vol, brain, tumor, zooms = tg.load_case(case_id)
    if args.source == "expert":
        if tumor is None:
            print("ERROR: this case has no expert segmentation (_seg file).")
            sys.exit(1)
        mask = tumor.astype(bool)
        prob = None
    else:
        model = tg.load_trained_model()
        prob  = tg.predict_case(model, vol, brain)
        mask  = prob > tg.PROB_THRESHOLD

    voxel_ml = float(np.prod(zooms)) / 1000.0
    v0 = mask.sum() * voxel_ml
    if v0 == 0:
        print("  No tumor core in this mask; nothing to project.")
        sys.exit(0)

    sgr = size_group_sgr(v0)
    print(f"  Current tumor-core volume V0: {v0:.1f} mL "
          f"(size group median SGR: {sgr} %/day)\n")

    # time-course table
    header = f"  {'Model':<24}" + "".join(f"{f'day {d}':>10}" for d in REPORT_DAYS)
    print(header + f"{f'day {t:g}':>10}")
    print("  " + "-" * (len(header) - 2 + 10))
    projected = {}
    for name, fn, _ in MODELS:
        vals = [fn(v0, d) for d in REPORT_DAYS] + [fn(v0, t)]
        projected[name] = vals[-1]
        print(f"  {name:<24}" + "".join(f"{v:>10.1f}" for v in vals))
    print("  (volumes in mL)\n")
    if v0 >= GOMPERTZ_K_ML:
        print(f"  Note: V0 >= K ({GOMPERTZ_K_ML} mL), so the Gompertz fit predicts no further growth.\n")

    # projected masks
    grown = {}
    for name, _, _ in MODELS:
        grown[name], capped = grow_mask(mask, brain, zooms, projected[name])
        if capped:
            print(f"  ⚠ {name}: projected volume exceeds available brain voxels; drawing is capped.")

    # display slice = largest current tumor area
    z = int(np.argmax(mask.sum(axis=(0, 1))))
    gray = tg.get_slice(vol[0], z)                 # T1ce
    mask2d = tg.get_slice(mask, z)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.axis("off")

    # 1. original
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original MRI (T1ce)", color="white", fontsize=12, pad=10)

    # 2. detection / segmentation
    axes[1].imshow(gray, cmap="gray")
    rgba = np.zeros((*gray.shape, 4))
    if prob is not None:
        p2d = tg.get_slice(prob, z)
        m = p2d > tg.PROB_THRESHOLD
        rgba[m, 0] = 1.0
        rgba[m, 3] = np.clip(p2d[m], 0, 1) * 0.65
        title2 = "Detection Overlay (U-Net)"
    else:
        rgba[mask2d, 0] = 1.0
        rgba[mask2d, 3] = 0.55
        title2 = "Tumor Core (expert segmentation)"
    axes[1].imshow(rgba)
    axes[1].set_title(title2, color="white", fontsize=12, pad=10)

    # 3. projected growth
    axes[2].imshow(gray, cmap="gray")
    handles = [mlines.Line2D([], [], color="white", lw=2, label=f"Now: {v0:.1f} mL")]
    for name, _, color in MODELS:
        draw_contours(axes[2], tg.get_slice(grown[name], z), color, lw=2)
        handles.append(mlines.Line2D([], [], color=color, lw=2,
                                     label=f"{name}: {projected[name]:.1f} mL"))
    draw_contours(axes[2], mask2d, "white", lw=2)
    leg = axes[2].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.2),
                         facecolor="#161b22", edgecolor="#30363d", fontsize=9)
    for txt in leg.get_texts():
        txt.set_color("white")
    title3 = f"Projected Growth at {t:g} Days" + (" (extrapolated)" if extrapolated else "")
    axes[2].set_title(title3, color="white", fontsize=12, pad=10)

    fig.text(0.5, -0.06,
             "Projection: population models for untreated glioblastoma (Stensjøen et al. 2015, "
             "Neuro-Oncology). Individual growth varies widely; the paper observed 14–98 days.\n"
             "Volumes come from the models; the even outward expansion is a display choice, "
             "not a predicted shape.   ⚠ Research / educational tool only. NOT for clinical use.",
             ha="center", color="#f0883e", fontsize=9)
    plt.suptitle(f"{case_id}: Tumor Core and Projected Growth",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(tg.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(tg.OUTPUT_DIR,
                            f"{case_id}_growth_{t:g}d_{args.source}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print(f"  ✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
