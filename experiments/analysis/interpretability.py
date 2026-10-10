"""
Part 1: Interpretability -- CBAM spatial attention + Grad-CAM visualization
for correctly-classified MIT67 / UC Merced test images using the finalized
CG-CBAT checkpoints.

Pure inference: no training, no architecture changes. The CBAM spatial
attention map is captured via a forward hook on `model.cbam_final.spatial_att`
(no modification to model.py or its default forward() behaviour). The
Grad-CAM-style map uses `torch.autograd.grad` on the predicted class logit
w.r.t. `gated_feat` (the final fused+gated feature map returned by
`forward(x, return_attention=True)`), global-average-pooling the gradient per
channel to get Grad-CAM weights, weighted-summing the feature channels,
ReLU-ing, and normalizing -- standard Grad-CAM, applied to the LAST feature
map before global pooling (the most semantically relevant choice, and the
same map CBAM's own spatial attention acts just upstream of).

Outputs, per example, to results/interpretability/:
    <dataset>_<class>_<index>_original.png
    <dataset>_<class>_<index>_cbam.png
    <dataset>_<class>_<index>_gradcam.png
plus a small JSON of per-example stats (raw cbam attention min/max, confidence)
used to write the honest summary.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

OUT_DIR = common.RESULTS_DIR / "interpretability"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (dataset, class_name, pick_nth_correct_image) -- pick_nth lets us avoid
# always grabbing image #0 of a class if we want variety; 0-indexed among
# CORRECTLY classified images of that class only.
EXAMPLES = [
    ("ucmerced", "parkinglot", 0),
    ("ucmerced", "golfcourse", 0),
    ("ucmerced", "forest", 0),
    ("ucmerced", "denseresidential", 0),
    ("mit67", "bookstore", 0),
    ("mit67", "kitchen", 0),
    ("mit67", "corridor", 0),
    ("mit67", "airport_inside", 0),
]

IMG_SIZE = 224


def jet_colormap(x: np.ndarray) -> np.ndarray:
    """Manual jet colormap (no matplotlib dependency available in venv).
    x: float array in [0,1], any shape. Returns same shape + (3,), uint8."""
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(np.minimum(4 * x - 1.5, -4 * x + 4.5), 0, 1)
    g = np.clip(np.minimum(4 * x - 0.5, -4 * x + 3.5), 0, 1)
    b = np.clip(np.minimum(4 * x + 0.5, -4 * x + 2.5), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def overlay_heatmap(orig_rgb_uint8: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    colored = jet_colormap(heatmap_01).astype(np.float32)
    orig = orig_rgb_uint8.astype(np.float32)
    blended = (1 - alpha) * orig + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


def upsample_to_224(t: torch.Tensor) -> np.ndarray:
    """t: (1,1,h,w) tensor -> (224,224) numpy array, bilinear upsample."""
    up = F.interpolate(t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    return up.squeeze().detach().cpu().numpy()


def find_correct_image(model, class_names, dataset_name, class_name, nth, device):
    """Scans the class's test images (in dataset order) and returns the
    (nth-counting-from-0) one the model classifies correctly, along with its
    file path, ground-truth index, predicted index, and softmax confidence."""
    from torchvision import datasets, transforms
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    class_dir = common.DATASETS_DIR / dataset_name / "test" / class_name
    if not class_dir.is_dir():
        raise RuntimeError(f"class dir not found: {class_dir}")
    target_idx = class_names.index(class_name)
    files = sorted(class_dir.iterdir())
    found = 0
    for fp in files:
        try:
            img = Image.open(fp).convert("RGB")
        except Exception:
            continue
        tensor = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
        pred = logits.argmax(dim=1).item()
        conf = F.softmax(logits, dim=1)[0, pred].item()
        if pred == target_idx:
            if found == nth:
                return fp, target_idx, pred, conf
            found += 1
    raise RuntimeError(f"Could not find {nth+1} correctly-classified images for "
                        f"{dataset_name}/{class_name} (found {found})")


def main():
    device = common.get_device()
    print(f"Using device: {device}")

    stats = []

    # Group examples by dataset so we only load each CG-CBAT model once.
    by_dataset = {}
    for dataset_name, class_name, nth in EXAMPLES:
        by_dataset.setdefault(dataset_name, []).append((class_name, nth))

    for dataset_name, items in by_dataset.items():
        print(f"\n=== {dataset_name} ===")
        model, class_names = common.load_cgcbat_model(dataset_name, img_size=IMG_SIZE, device=device)

        # Forward hook to capture CBAM final-stage spatial attention map
        # (B,1,7,7), produced right before it's multiplied into the feature
        # map inside CBAM.forward(). No change to model.py.
        captured = {}

        def hook(module, inp, out):
            captured["spatial_att"] = out.detach()

        handle = model.cbam_final.spatial_att.register_forward_hook(hook)

        for class_name, nth in items:
            print(f"  finding correctly-classified '{class_name}' image (nth={nth})...")
            fp, target_idx, pred_idx, conf = find_correct_image(
                model, class_names, dataset_name, class_name, nth, device
            )
            print(f"    found: {fp.name} pred={class_names[pred_idx]} conf={conf:.4f}")

            # Reload tensor for the gradient-enabled forward pass.
            from torchvision import transforms
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            eval_tf = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            pil_img = Image.open(fp).convert("RGB")
            orig_resized = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            orig_np = np.array(orig_resized)  # (224,224,3) uint8

            tensor = eval_tf(pil_img).unsqueeze(0).to(device)

            captured.clear()
            logits, gated_feat, gate = model(tensor, return_attention=True)
            pred = logits.argmax(dim=1)
            assert pred.item() == pred_idx, "prediction changed between passes (unexpected)"

            # --- CBAM spatial attention map ---
            cbam_map = captured["spatial_att"]  # (1,1,7,7), already sigmoid in [0,1]
            cbam_raw_min, cbam_raw_max = cbam_map.min().item(), cbam_map.max().item()
            cbam_up = upsample_to_224(cbam_map)
            cbam_norm = minmax_norm(cbam_up)

            # --- Grad-CAM on gated_feat w.r.t. predicted logit ---
            selected_logit = logits[0, pred_idx]
            grad = torch.autograd.grad(selected_logit, gated_feat, retain_graph=False)[0]  # (1,2048,7,7)
            weights = grad.mean(dim=(2, 3), keepdim=True)  # (1,2048,1,1)
            cam = (weights * gated_feat).sum(dim=1, keepdim=True)  # (1,1,7,7)
            cam = F.relu(cam)
            cam_up = upsample_to_224(cam)
            cam_norm = minmax_norm(cam_up)

            # --- Save PNGs ---
            tag = f"{dataset_name}_{class_name}_{nth}"
            Image.fromarray(orig_np).save(OUT_DIR / f"{tag}_original.png")
            Image.fromarray(overlay_heatmap(orig_np, cbam_norm, alpha=0.4)).save(OUT_DIR / f"{tag}_cbam.png")
            Image.fromarray(overlay_heatmap(orig_np, cam_norm, alpha=0.4)).save(OUT_DIR / f"{tag}_gradcam.png")

            stats.append({
                "dataset": dataset_name,
                "class": class_name,
                "index": nth,
                "file": str(fp),
                "predicted_class": class_names[pred_idx],
                "ground_truth_class": class_names[target_idx],
                "confidence": conf,
                "cbam_raw_attention_min": cbam_raw_min,
                "cbam_raw_attention_max": cbam_raw_max,
                "cam_raw_min": float(cam_up.min()),
                "cam_raw_max": float(cam_up.max()),
            })
            print(f"    saved {tag}_original/cbam/gradcam.png "
                  f"(cbam raw range [{cbam_raw_min:.4f},{cbam_raw_max:.4f}])")

        handle.remove()

    with open(OUT_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved stats to {OUT_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()
