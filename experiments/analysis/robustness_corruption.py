"""
Part 2: Robustness -- corruption testing (Gaussian blur, Gaussian noise) for
CG-CBAT vs. plain ResNet50 on Intel Image (TIE case) and MIT67 (WIN case).

Pure inference: no training, no architecture changes. Reuses `evaluate()`
from experiments/baseline/train_eval.py unchanged -- the only thing that
varies across runs is the test-set DataLoader's transform pipeline (adding a
corruption step), never the model or the evaluation logic itself.

Corruption severities:
    - "clean": identical to train_eval.py's normal eval_tf (sanity check --
      must reproduce the known existing accuracies exactly).
    - "gaussian_blur" sigma in {1.0, 2.0, 3.0}: torchvision GaussianBlur
      applied to the raw (0-1 range) image tensor, BEFORE normalization
      (i.e. blurring in pixel space, like real optical/motion blur would).
    - "gaussian_noise" std in {0.1, 0.25, 0.5}: additive IID Gaussian noise
      applied to the NORMALIZED tensor (so std is directly comparable to the
      ~unit per-channel variance of normalized inputs), AFTER normalization.
"""

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

te = common.te

IMG_SIZE = 224
BATCH_SIZE = 32
WORKERS = 4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

BLUR_SIGMAS = [1.0, 2.0, 3.0]
NOISE_STDS = [0.1, 0.25, 0.5]

DATASETS_TO_TEST = ["intel_image", "mit67"]

OUT_JSON = common.RESULTS_DIR / "robustness_corruption.json"


class AddGaussianNoise:
    """Adds IID N(0, std^2) noise to a normalized tensor. Picklable callable
    (not a lambda) so DataLoader workers (num_workers>0) can use it."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, tensor):
        if self.std == 0.0:
            return tensor
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self):
        return f"AddGaussianNoise(std={self.std})"


def build_transform(corruption_type: str, severity):
    """Returns a torchvision transform matching train_eval.py's eval_tf,
    with an added corruption step."""
    if corruption_type == "clean":
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    if corruption_type == "gaussian_blur":
        sigma = float(severity)
        # kernel size must be odd and large enough to realize the given
        # sigma (common rule of thumb: >= 6*sigma, rounded up to odd).
        k = int(2 * round(3 * sigma) + 1)
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=k, sigma=sigma),
            transforms.Normalize(MEAN, STD),
        ])
    if corruption_type == "gaussian_noise":
        std = float(severity)
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            AddGaussianNoise(std),
        ])
    raise ValueError(corruption_type)


def build_corrupted_test_loader(dataset_name: str, corruption_type: str, severity):
    tf = build_transform(corruption_type, severity)
    ds = datasets.ImageFolder(common.DATASETS_DIR / dataset_name / "test", transform=tf)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    return loader, ds.classes


def main():
    device = common.get_device()
    print(f"Using device: {device}")

    conditions = [("clean", 0)]
    for s in BLUR_SIGMAS:
        conditions.append(("gaussian_blur", s))
    for s in NOISE_STDS:
        conditions.append(("gaussian_noise", s))

    results = {}  # dataset -> model_name -> corruption_type -> severity(str) -> accuracy

    for dataset_name in DATASETS_TO_TEST:
        results[dataset_name] = {"cg_cbat": {}, "resnet50": {}}
        print(f"\n{'='*60}\n{dataset_name}\n{'='*60}")

        cgcbat_model, class_names = common.load_cgcbat_model(dataset_name, img_size=IMG_SIZE, device=device)
        resnet_model, class_names2 = common.load_resnet50_model(dataset_name, img_size=IMG_SIZE, device=device)
        assert class_names == class_names2

        for corruption_type, severity in conditions:
            print(f"\n-- {dataset_name} / {corruption_type} severity={severity} --")
            t0 = time.time()
            loader, loader_classes = build_corrupted_test_loader(dataset_name, corruption_type, severity)
            assert loader_classes == class_names

            for model_name, model in (("cg_cbat", cgcbat_model), ("resnet50", resnet_model)):
                r = te.evaluate(model, loader, class_names, device)
                acc = r["accuracy"]
                results[dataset_name][model_name].setdefault(corruption_type, {})[str(severity)] = acc
                print(f"  {model_name:10s} accuracy={acc:.4f}")

            print(f"  (took {time.time()-t0:.1f}s)")

        del cgcbat_model, resnet_model

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {OUT_JSON}")


if __name__ == "__main__":
    main()
