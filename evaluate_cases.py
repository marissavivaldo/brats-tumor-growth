"""
evaluate_cases.py
-----------------
Add-on to tumor_growth_brats.py and growth_projection.py (neither is modified).

For every held-out case (val_cases.txt), compares the U-Net's tumor-core
volume with the expert segmentation, and checks how that difference carries
into the growth projection. Writes one row per case to results/eval_volumes.csv
and prints a summary.

Columns
-------
case                    BraTS case ID
expert_ml, unet_ml      tumor-core volume (labels 1+4) in mL
diff_ml, diff_pct       U-Net minus expert (negative = U-Net underestimates)
dice_3d                 3D Dice between U-Net and expert masks
expert_group,           size group from Stensjoen et al. 2015
unet_group              (small < 3.88 mL, medium < 36.88 mL, large >= 36.88 mL)
group_match             1 if both masks fall in the same size group
expert_radial_ml,       linear radial projection at --days (default 30)
unet_radial_ml
proj_diff_ml            U-Net minus expert, projected

Usage
-----
    python evaluate_cases.py               # all held-out cases, 30 days
    python evaluate_cases.py --days 60
    python evaluate_cases.py --out results/eval_after_full_training.csv
"""

import os
import sys
import csv
import argparse
import numpy as np

import tumor_growth_brats as tg
import growth_projection as gp


def size_group(v_ml):
    if v_ml <= 0:
        return "none"
    if v_ml < gp.SIZE_GROUPS[0][0]:
        return "small"
    if v_ml < gp.SIZE_GROUPS[1][0]:
        return "medium"
    return "large"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=30)
    parser.add_argument("--out",  default=os.path.join(tg.OUTPUT_DIR, "eval_volumes.csv"))
    args = parser.parse_args()

    if not os.path.exists(tg.VAL_LIST_PATH):
        print("ERROR: val_cases.txt not found. Run tumor_growth_brats.py --retrain first.")
        sys.exit(1)
    with open(tg.VAL_LIST_PATH) as f:
        cases = [line.strip() for line in f if line.strip()]

    model = tg.load_trained_model()
    print(f"  Evaluating {len(cases)} held-out case(s); projection at {args.days:g} days\n")
    print(f"  {'Case':<17}{'Expert':>8}{'U-Net':>8}{'Diff':>8}{'Diff %':>8}{'Dice':>7}  Groups")
    print("  " + "-" * 72)

    rows = []
    for case_id in cases:
        vol, brain, tumor, zooms = tg.load_case(case_id)
        if tumor is None:
            print(f"  {case_id:<17} skipped (no expert segmentation)")
            continue
        expert = tumor.astype(bool)
        unet   = tg.predict_case(model, vol, brain) > tg.PROB_THRESHOLD

        voxel_ml  = float(np.prod(zooms)) / 1000.0
        expert_ml = expert.sum() * voxel_ml
        unet_ml   = unet.sum() * voxel_ml
        diff_ml   = unet_ml - expert_ml
        diff_pct  = diff_ml / expert_ml * 100 if expert_ml > 0 else float("nan")
        denom     = expert.sum() + unet.sum()
        dice      = 1.0 if denom == 0 else 2 * (expert & unet).sum() / denom

        eg, ug = size_group(expert_ml), size_group(unet_ml)
        e_proj = gp.v_radial(expert_ml, args.days) if expert_ml > 0 else 0.0
        u_proj = gp.v_radial(unet_ml, args.days) if unet_ml > 0 else 0.0

        rows.append({
            "case": case_id,
            "expert_ml": round(expert_ml, 2), "unet_ml": round(unet_ml, 2),
            "diff_ml": round(diff_ml, 2), "diff_pct": round(diff_pct, 1),
            "dice_3d": round(dice, 3),
            "expert_group": eg, "unet_group": ug, "group_match": int(eg == ug),
            "expert_radial_ml": round(e_proj, 2), "unet_radial_ml": round(u_proj, 2),
            "proj_diff_ml": round(u_proj - e_proj, 2),
        })
        flag = "" if eg == ug else "  ← differs"
        print(f"  {case_id:<17}{expert_ml:>8.1f}{unet_ml:>8.1f}{diff_ml:>8.1f}"
              f"{diff_pct:>7.0f}%{dice:>7.3f}  {eg}/{ug}{flag}")

    if not rows:
        print("  No cases evaluated.")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    d     = np.array([r["diff_ml"] for r in rows])
    pct   = np.array([r["diff_pct"] for r in rows if not np.isnan(r["diff_pct"])])
    dices = np.array([r["dice_3d"] for r in rows])
    match = sum(r["group_match"] for r in rows)

    print("\n  Summary (U-Net vs expert, tumor core)")
    print(f"    Cases:                         {len(rows)}")
    print(f"    Mean 3D Dice:                  {dices.mean():.3f}  (median {np.median(dices):.3f})")
    print(f"    Mean difference (bias):        {d.mean():+.1f} mL")
    print(f"    Mean absolute difference:      {np.abs(d).mean():.1f} mL")
    if pct.size:
        print(f"    Median absolute difference:    {np.median(np.abs(pct)):.0f}%")
    print(f"    Same size group:               {match}/{len(rows)}")
    print(f"\n  ✓ Saved → {args.out}")


if __name__ == "__main__":
    main()
