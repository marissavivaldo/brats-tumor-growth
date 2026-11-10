"""
tumor_growth_brats.py
---------------------
Copy of tumor_unet_real_data.py, adapted to the BraTS 2021 glioblastoma
dataset. This is step 1 of the growth-projection project: segment the tumor
core (the same region Stensjoen et al. 2015 measured) and measure its volume
in mL. Growth projection is NOT added yet.

What changed from tumor_unet_real_data.py
-----------------------------------------
- Data: BraTS 2021 .nii.gz volumes (data/BraTS2021_xxxxx/) instead of LGG .tif slices
- Input channels: T1ce, T2, FLAIR (3 channels, so the U-Net is unchanged)
- Target: tumor core = necrotic core (label 1) + enhancing tumor (label 4).
  This matches the paper's tumor volume: contrast-enhancing part + central
  non-enhancing part. Edema (label 2) is excluded, as in the paper.
- Slices are extracted once into slice_cache/ so training doesn't re-read
  the full 3D volumes every epoch
- Validation split by case (patient); validation slices are not flipped
- Inference runs on a whole held-out BraTS case and reports tumor-core
  volume in mL, plus 3D Dice against the expert segmentation
- Detection threshold 0.5 (the same threshold validation Dice is scored at)
- Brain mask comes from the scan itself (BraTS is skull-stripped) instead of
  a fixed ellipse; no Gaussian blur on the probability map
- After training, the BEST saved weights are loaded for inference
Unchanged: U-Net architecture, loss, optimizer, scheduler, training loop,
figure style.

Requirements
------------
    pip install torch numpy nibabel scikit-image matplotlib

Usage
-----
    # Quick test on 40 cases, 2 epochs (checks everything runs):
    python tumor_growth_brats.py --retrain --max-cases 40 --epochs 2

    # Full training (all cases):
    python tumor_growth_brats.py --retrain

    # Run the trained model on held-out validation cases:
    python tumor_growth_brats.py

    # Or on one specific case:
    python tumor_growth_brats.py --case BraTS2021_00000
"""

import sys
print("PYTHON VERSION:", sys.version)

import os
import random
import argparse
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import measure, morphology
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0.  CONFIG
# ──────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR       = os.path.join(SCRIPT_DIR, "data")
CACHE_DIR      = os.path.join(SCRIPT_DIR, "slice_cache")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "results")
MODEL_PATH     = os.path.join(SCRIPT_DIR, "brats_tumor_core_weights.pth")
VAL_LIST_PATH  = os.path.join(SCRIPT_DIR, "val_cases.txt")

MODALITIES     = ("t1ce", "t2", "flair")   # -> 3 input channels
TUMOR_LABELS   = (1, 4)                    # tumor core = necrotic core + enhancing
TUMOR_SLICES_PER_CASE = 8                  # evenly spaced slices containing tumor
EMPTY_SLICES_PER_CASE = 2                  # random brain slices with no tumor

PROB_THRESHOLD = 0.5
MIN_REGION_PX  = 20
MAX_REGIONS    = 3
TRAIN_EPOCHS   = 30
BATCH_SIZE     = 4
LEARNING_RATE  = 1e-3
VAL_FRACTION   = 0.15   # held-out fraction of PATIENTS (not slices) for validation


# ──────────────────────────────────────────────
# 1.  DEVICE
# ──────────────────────────────────────────────
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = pick_device()
print(f"\n{'='*60}")
print(f"  BraTS Tumor-Core U-Net")
print(f"  Device: {DEVICE}")
print(f"{'='*60}\n")


# ──────────────────────────────────────────────
# 2.  U-NET ARCHITECTURE  (unchanged)
# ──────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        self.pool  = nn.MaxPool2d(2, 2)

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, 2))
            self.ups.append(DoubleConv(f * 2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            s = skips[i // 2]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, size=s.shape[2:])
            x = torch.cat([s, x], dim=1)
            x = self.ups[i + 1](x)

        return torch.sigmoid(self.final(x))


# ──────────────────────────────────────────────
# 3.  BRATS DATA  (new)
# ──────────────────────────────────────────────
def list_cases(data_dir=DATA_DIR):
    if not os.path.isdir(data_dir):
        return []
    return sorted(d for d in os.listdir(data_dir)
                  if d.startswith("BraTS2021_") and os.path.isdir(os.path.join(data_dir, d)))


def load_case(case_id):
    """
    Returns
      vol    (3, X, Y, Z) float32, each sequence z-scored over brain voxels
      brain  (X, Y, Z) bool, nonzero voxels (BraTS is skull-stripped)
      tumor  (X, Y, Z) uint8 tumor-core mask, or None if no _seg file
      zooms  voxel size in mm (read from the file header)
    """
    case_dir = os.path.join(DATA_DIR, case_id)
    channels = []
    for m in MODALITIES:
        img = nib.load(os.path.join(case_dir, f"{case_id}_{m}.nii.gz"))
        channels.append(img.get_fdata(dtype=np.float32))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])

    vol = np.stack(channels, axis=0)
    brain = np.any(vol > 0, axis=0)
    for c in range(vol.shape[0]):
        vals = vol[c][brain]
        vol[c] = np.where(brain, (vol[c] - vals.mean()) / (vals.std() + 1e-8), 0.0)

    seg_path = os.path.join(case_dir, f"{case_id}_seg.nii.gz")
    tumor = None
    if os.path.exists(seg_path):
        seg = np.asarray(nib.load(seg_path).dataobj)
        tumor = np.isin(seg, TUMOR_LABELS).astype(np.uint8)
    return vol, brain, tumor, zooms


def get_slice(arr, z):
    """Axial slice z of (X,Y,Z) or (C,X,Y,Z), rotated for display. Used for
    training AND inference so orientation is always the same."""
    return np.ascontiguousarray(np.rot90(arr[..., z], axes=(-2, -1)))


def cache_paths(case_id):
    return (os.path.join(CACHE_DIR, f"{case_id}_img.npy"),
            os.path.join(CACHE_DIR, f"{case_id}_mask.npy"))


def build_slice_cache(cases, seed=42):
    """Extracts selected 2D slices from each 3D case once and saves them.
    Cases already in the cache are skipped, so this can be interrupted."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    todo = [c for c in cases if not os.path.exists(cache_paths(c)[1])]
    if not todo:
        return
    print(f"  Extracting slices from {len(todo)} case(s) into slice_cache/ "
          f"(one-time step)...")
    rng = np.random.default_rng(seed)
    for i, case_id in enumerate(todo, 1):
        vol, brain, tumor, _ = load_case(case_id)
        if tumor is None:
            raise RuntimeError(f"{case_id} has no _seg.nii.gz file; can't train on it.")
        tumor_area = tumor.sum(axis=(0, 1))
        brain_area = brain.sum(axis=(0, 1))
        tumor_z = np.where(tumor_area > 0)[0]
        empty_z = np.where((tumor_area == 0) & (brain_area > 0.05 * brain_area.max()))[0]

        pick = []
        if len(tumor_z):
            k = min(TUMOR_SLICES_PER_CASE, len(tumor_z))
            pick += [int(z) for z in tumor_z[np.linspace(0, len(tumor_z) - 1, k).round().astype(int)]]
        if len(empty_z):
            k = min(EMPTY_SLICES_PER_CASE, len(empty_z))
            pick += [int(z) for z in rng.choice(empty_z, size=k, replace=False)]

        h, w = get_slice(tumor, 0).shape
        imgs  = (np.stack([get_slice(vol, z) for z in pick]).astype(np.float16)
                 if pick else np.zeros((0, 3, h, w), np.float16))
        masks = (np.stack([get_slice(tumor, z) for z in pick]).astype(np.uint8)
                 if pick else np.zeros((0, h, w), np.uint8))
        img_path, mask_path = cache_paths(case_id)
        np.save(img_path, imgs)
        np.save(mask_path, masks)   # saved last: its presence marks the case as done
        if i % 50 == 0 or i == len(todo):
            print(f"    {i}/{len(todo)} cases")
    print()


def split_cases(cases, val_fraction=VAL_FRACTION, seed=42):
    """Splits by CASE (patient), so no patient appears in both sets."""
    ids = list(cases)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


class BraTSSliceDataset(Dataset):
    def __init__(self, cases, augment=True):
        self.items = []
        for c in cases:
            n = np.load(cache_paths(c)[1], mmap_mode="r").shape[0]
            self.items += [(c, i) for i in range(n)]
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        case_id, i = self.items[idx]
        img_path, mask_path = cache_paths(case_id)
        arr  = np.load(img_path,  mmap_mode="r")[i].astype(np.float32)   # (3,H,W)
        mask = np.load(mask_path, mmap_mode="r")[i].astype(np.float32)   # (H,W)

        # light augmentation (same flips as before), training only
        if self.augment:
            if random.random() > 0.5:
                arr, mask = arr[:, :, ::-1], mask[:, ::-1]
            if random.random() > 0.5:
                arr, mask = arr[:, ::-1, :], mask[::-1, :]

        return torch.from_numpy(arr.copy()), torch.from_numpy(mask.copy()).unsqueeze(0)


@torch.no_grad()
def dice_score(preds, targets, smooth=1.0):
    p = preds.view(preds.size(0), -1)
    t = targets.view(targets.size(0), -1)
    intersection = (p * t).sum(dim=1)
    union = p.sum(dim=1) + t.sum(dim=1)
    return ((2 * intersection + smooth) / (union + smooth)).mean().item()


# ──────────────────────────────────────────────
# 4.  LOSS FUNCTION  (unchanged)
# ──────────────────────────────────────────────
class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce  = self.bce(pred, target)
        p    = pred.view(-1)
        t    = target.view(-1)
        dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
        return bce + dice


# ──────────────────────────────────────────────
# 5.  TRAINING  (same loop; BraTS data)
# ──────────────────────────────────────────────
def train_model(cases, epochs=TRAIN_EPOCHS):
    print("  Training U-Net on BraTS 2021 (tumor core)...")
    print(f"  (Weights saved to {os.path.basename(MODEL_PATH)})\n")

    build_slice_cache(cases)
    train_cases, val_cases = split_cases(cases)
    with open(VAL_LIST_PATH, "w") as f:
        f.write("\n".join(val_cases) + "\n")

    train_dataset = BraTSSliceDataset(train_cases, augment=True)
    val_dataset   = BraTSSliceDataset(val_cases, augment=False)
    print(f"  Cases  -- train: {len(train_cases)}, val: {len(val_cases)} "
          f"(split by patient; val list saved to {os.path.basename(VAL_LIST_PATH)})")
    print(f"  Slices -- train: {len(train_dataset)}, val: {len(val_dataset)}\n")

    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader    = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = UNet(in_channels=3, out_channels=1).to(DEVICE)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  {'Epoch':<8} {'Loss':<12} {'Val Dice':<10} {'Status'}")
    print(f"  {'-'*8} {'-'*11} {'-'*9} {'-'*20}")

    best_val_dice = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_dice_total = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                preds = model(imgs)
                val_dice_total += dice_score((preds > 0.5).float(), masks)
        val_dice = val_dice_total / len(val_loader)

        marker = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), MODEL_PATH)
            marker = " ◀ best"

        print(f"  {epoch:<8} {avg_loss:<12.4f} {val_dice:<10.4f}{marker}")

    print(f"\n  ✓ Training complete. Best val Dice: {best_val_dice:.4f}")
    print(f"  ✓ Weights saved → {MODEL_PATH}\n")


def load_trained_model():
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ──────────────────────────────────────────────
# 6.  INFERENCE ON A WHOLE CASE  (new)
# ──────────────────────────────────────────────
@torch.no_grad()
def predict_case(model, vol, brain, batch=16):
    """Runs the 2D U-Net on every brain slice; returns a 3D probability map (X,Y,Z)."""
    prob = np.zeros(brain.shape, dtype=np.float32)
    zs = [z for z in range(brain.shape[-1]) if brain[..., z].any()]
    for start in range(0, len(zs), batch):
        batch_z = zs[start:start + batch]
        x = torch.from_numpy(np.stack([get_slice(vol, z) for z in batch_z])).to(DEVICE)
        p = model(x).squeeze(1).cpu().numpy()
        for z, pz in zip(batch_z, p):
            prob[..., z] = np.rot90(pz, k=-1)   # undo get_slice rotation
    return prob * brain


def find_regions(prob):
    binary = prob > PROB_THRESHOLD
    binary = morphology.remove_small_objects(binary, min_size=MIN_REGION_PX)
    binary = morphology.remove_small_holes(binary, area_threshold=40)
    labeled = measure.label(binary)
    props   = measure.regionprops(labeled, intensity_image=prob)

    regions = []
    for r in props:
        if r.area < MIN_REGION_PX:
            continue
        regions.append({
            "centroid":   r.centroid,
            "area":       int(r.area),
            "confidence": float(r.mean_intensity),
            "max_conf":   float(r.max_intensity),
            "bbox":       r.bbox,
            "diameter":   float(r.equivalent_diameter),
        })
    regions.sort(key=lambda d: d["max_conf"], reverse=True)
    return regions[:MAX_REGIONS]


# ──────────────────────────────────────────────
# 7.  VISUALISATION  (unchanged)
# ──────────────────────────────────────────────
def visualise(name, gray, prob, regions, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        for s in ax.spines.values():
            s.set_edgecolor("#30363d")

    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Original MRI", color="white", fontsize=12, pad=10)
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    rgba      = np.zeros((*prob.shape, 4))
    mask      = prob > PROB_THRESHOLD
    rgba[mask, 0] = 1.0
    rgba[mask, 3] = np.clip(prob[mask], 0, 1) * 0.65
    axes[1].imshow(rgba)

    colors = ["#ff4444", "#ff9900", "#ffff00"]
    for i, r in enumerate(regions):
        c        = colors[i % len(colors)]
        r0, c0, r1, c1 = r["bbox"]
        axes[1].add_patch(mpatches.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            linewidth=3 if i == 0 else 2,
            edgecolor=c, facecolor="none", linestyle="--"))
    axes[1].set_title("Detection Overlay", color="white", fontsize=12, pad=10)
    axes[1].axis("off")

    fig.text(0.5, 0.02,
             "⚠  Research / educational tool only. NOT for clinical diagnosis. "
             "Always consult a medical professional.",
             ha="center", color="#f0883e", fontsize=9)
    plt.suptitle("Trained U-Net Brain Tumor Detection",
                 color="white", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0d1117", edgecolor="none")
    plt.close()
    print(f"    ✓ Saved → {out_path}")


# ──────────────────────────────────────────────
# 8.  PROCESS ONE CASE  (new: whole 3D case, volume in mL)
# ──────────────────────────────────────────────
def process_case(model, case_id):
    print(f"\n  Analysing: {case_id}")
    vol, brain, tumor, zooms = load_case(case_id)
    prob = predict_case(model, vol, brain)
    pred = prob > PROB_THRESHOLD

    voxel_ml = float(np.prod(zooms)) / 1000.0          # mm^3 -> mL
    pred_ml  = pred.sum() * voxel_ml
    print(f"    Voxel size: {zooms[0]:.2f} x {zooms[1]:.2f} x {zooms[2]:.2f} mm")
    print(f"    Predicted tumor-core volume:     {pred_ml:6.1f} mL")
    if tumor is not None:
        true_ml = tumor.sum() * voxel_ml
        denom = pred.sum() + tumor.sum()
        dice = 1.0 if denom == 0 else 2 * (pred & tumor.astype(bool)).sum() / denom
        print(f"    Expert-segmented volume:         {true_ml:6.1f} mL")
        print(f"    3D Dice vs expert segmentation:  {dice:.3f}")

    # show the slice with the largest predicted tumor area
    areas = pred.sum(axis=(0, 1))
    z = int(np.argmax(areas)) if areas.max() > 0 else int(np.argmax(brain.sum(axis=(0, 1))))
    gray   = get_slice(vol[0], z)        # T1ce
    prob2d = get_slice(prob, z)
    regions = find_regions(prob2d)
    print(f"    Displayed slice: z = {z}")
    if regions:
        for i, r in enumerate(regions, 1):
            cy2, cx2 = r["centroid"]
            print(f"    #{i}  {r['max_conf']:.1%}  @ ({cx2:.0f},{cy2:.0f})  "
                  f"⌀{r['diameter'] * zooms[0]:.1f} mm")
    else:
        print("    No tumor regions above threshold on this slice.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"{case_id}_result.png")
    visualise(case_id, gray, prob2d, regions, out_path)


# ──────────────────────────────────────────────
# 9.  MAIN
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",      help="Analyse one case, e.g. BraTS2021_00000")
    parser.add_argument("--retrain",   action="store_true",
                        help="Force retraining even if weights exist")
    parser.add_argument("--max-cases", type=int, default=None,
                        help="Use only the first N cases (for a quick test run)")
    parser.add_argument("--epochs",    type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--n-show",    type=int, default=3,
                        help="How many validation cases to analyse after training")
    args = parser.parse_args()

    cases = list_cases()
    if not cases:
        print(f"ERROR: no BraTS2021_* folders found in {DATA_DIR}")
        sys.exit(1)
    if args.max_cases:
        cases = cases[:args.max_cases]

    if not os.path.exists(MODEL_PATH) or args.retrain:
        train_model(cases, epochs=args.epochs)
    print(f"  ✓ Loading saved weights from {os.path.basename(MODEL_PATH)}")
    model = load_trained_model()

    if args.case:
        process_case(model, args.case)
    else:
        if not os.path.exists(VAL_LIST_PATH):
            print("ERROR: val_cases.txt not found. Run with --retrain first.")
            sys.exit(1)
        with open(VAL_LIST_PATH) as f:
            val_cases = [line.strip() for line in f if line.strip()]
        print(f"  Analysing {min(args.n_show, len(val_cases))} held-out validation case(s)")
        for c in val_cases[:args.n_show]:
            process_case(model, c)

    print("\n" + "=" * 60)
    print(f"  Done!  Results saved to:\n    {OUTPUT_DIR}")
    print("  ⚠  Research only — not for clinical use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
