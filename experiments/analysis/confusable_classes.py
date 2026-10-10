"""
Part 3: Robustness -- confusable-class stress test on SUN397.

Addresses the PhD guide's "shopping mall in airport" example directly:
SUN397's class list literally includes both "airport terminal" and
"shopping mall indoor" as separate classes (confirmed against
datasets/sun397/test/ folder names). This script runs full SUN397 test-set
inference for both the CG-CBAT and plain-ResNet50 finalized checkpoints,
builds a full confusion matrix for each, and extracts cross-confusion counts
for several visually-confusable class pairs.

Pure inference: no training, no architecture changes.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

IMG_SIZE = 224
BATCH_SIZE = 32
WORKERS = 4
DATASET = "sun397"

OUT_JSON = common.RESULTS_DIR / "robustness_confusable_classes.json"

# Confusable class pairs to report on (class names exactly as they appear in
# datasets/sun397/test/). First pair is the PhD guide's explicit example.
CONFUSABLE_PAIRS = [
    ("airport terminal", "shopping mall indoor"),
    ("forest broadleaf", "forest needleleaf"),
    ("conference room", "conference center"),
    ("kitchen", "kitchenette"),
]


@torch.no_grad()
def get_all_preds(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
    return np.array(all_labels), np.array(all_preds)


def pair_counts(cm, class_names, class_a, class_b):
    ia = class_names.index(class_a)
    ib = class_names.index(class_b)
    a_total = cm[ia].sum()
    b_total = cm[ib].sum()
    a_correct = cm[ia, ia]
    b_correct = cm[ib, ib]
    a_as_b = cm[ia, ib]  # true=a, predicted=b
    b_as_a = cm[ib, ia]  # true=b, predicted=a
    return {
        "class_a": class_a,
        "class_b": class_b,
        "class_a_test_count": int(a_total),
        "class_b_test_count": int(b_total),
        "class_a_correct": int(a_correct),
        "class_b_correct": int(b_correct),
        "a_misclassified_as_b": int(a_as_b),
        "b_misclassified_as_a": int(b_as_a),
        "a_misclassified_as_b_pct_of_a": float(a_as_b / a_total) if a_total else None,
        "b_misclassified_as_a_pct_of_b": float(b_as_a / b_total) if b_total else None,
    }


def main():
    device = common.get_device()
    print(f"Using device: {device}")

    ds, mean, std = common.build_eval_dataset(DATASET, img_size=IMG_SIZE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    class_names = ds.classes
    print(f"SUN397 test set: {len(ds)} images, {len(class_names)} classes")

    for pair in CONFUSABLE_PAIRS:
        for c in pair:
            assert c in class_names, f"class '{c}' not found in SUN397 class list"

    results = {}

    for model_key, loader_fn in (
        ("cg_cbat", common.load_cgcbat_model),
        ("resnet50", common.load_resnet50_model),
    ):
        print(f"\n=== {model_key} ===")
        model, cn = loader_fn(DATASET, img_size=IMG_SIZE, device=device)
        assert cn == class_names
        labels, preds = get_all_preds(model, loader, device)
        acc = float((labels == preds).mean())
        print(f"  overall accuracy: {acc:.4f}")

        cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))

        pairs_out = []
        for class_a, class_b in CONFUSABLE_PAIRS:
            pc = pair_counts(cm, class_names, class_a, class_b)
            pairs_out.append(pc)
            print(f"  [{class_a}] <-> [{class_b}]: "
                  f"{class_a}->{class_b}={pc['a_misclassified_as_b']}/{pc['class_a_test_count']}, "
                  f"{class_b}->{class_a}={pc['b_misclassified_as_a']}/{pc['class_b_test_count']}")

        results[model_key] = {
            "overall_accuracy": acc,
            "confusable_pairs": pairs_out,
        }
        del model

    results["class_names"] = class_names
    results["dataset"] = DATASET
    results["num_test_images"] = len(ds)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUT_JSON}")


if __name__ == "__main__":
    main()
