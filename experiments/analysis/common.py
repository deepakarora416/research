"""
Shared helpers for the Part 1/2/3 analysis scripts (interpretability,
corruption robustness, confusable-class stress test).

Pure inference/analysis code -- imports build_model/build_dataloaders/evaluate
from experiments/baseline/train_eval.py and reuses them rather than
reimplementing data loading or evaluation. Does NOT modify model.py or
train_eval.py in any way, and does not retrain anything.
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = REPO_ROOT / "experiments" / "baseline"
IMPROVED_DIR = REPO_ROOT / "experiments" / "improved"
DATASETS_DIR = REPO_ROOT / "datasets"
RESULTS_DIR = REPO_ROOT / "results"

sys.path.insert(0, str(BASELINE_DIR))
sys.path.insert(0, str(IMPROVED_DIR))

import train_eval as te  # noqa: E402

CUSTOM_MODEL_NAME = te.CUSTOM_MODEL_NAME  # "cg_cbat_resnet50"

# Dataset -> (cg_cbat checkpoint, resnet50 checkpoint)
CHECKPOINTS = {
    "mit67": (
        RESULTS_DIR / "cgcbat_v6_frozen_mit67_checkpoint.pt",
        RESULTS_DIR / "resnet50_8ep_mit67_checkpoint.pt",
    ),
    "ucmerced": (
        RESULTS_DIR / "cgcbat_v6_frozen_ucmerced_checkpoint.pt",
        RESULTS_DIR / "resnet50_8ep_ucmerced_checkpoint.pt",
    ),
    "intel_image": (
        RESULTS_DIR / "cgcbat_v8_val_intel_image_checkpoint.pt",
        RESULTS_DIR / "resnet50_8ep_intel_image_checkpoint.pt",
    ),
    "sun397": (
        RESULTS_DIR / "cgcbat_v6_frozen_sun397_checkpoint.pt",
        RESULTS_DIR / "resnet50_8ep_sun397_checkpoint.pt",
    ),
}


def get_device():
    return te.get_device()


def get_class_names(dataset: str):
    """Returns the ImageFolder class list for a dataset's test/ dir, in the
    same sorted order torchvision.datasets.ImageFolder uses (and therefore
    the same order the checkpoints' classifier output dims correspond to)."""
    from torchvision import datasets
    ds = datasets.ImageFolder(DATASETS_DIR / dataset / "test")
    return ds.classes


def load_cgcbat_model(dataset: str, img_size: int = 224, device=None, eval_mode=True):
    """Builds a fresh cg_cbat_resnet50 model and loads the FULL state dict
    from this dataset's finalized CG-CBAT checkpoint (checkpoint already
    contains the complete model -- backbone+CBAM+transformer+fusion+
    classifier -- so we load it directly, not via load_backbone_checkpoint).
    """
    class_names = get_class_names(dataset)
    num_classes = len(class_names)
    model = te.build_model(CUSTOM_MODEL_NAME, num_classes=num_classes, img_size=img_size)
    ckpt_path, _ = CHECKPOINTS[dataset]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()
    return model, class_names


def load_resnet50_model(dataset: str, img_size: int = 224, device=None, eval_mode=True):
    """Builds a plain timm resnet50 classifier and loads the finalized
    plain-ResNet50 checkpoint for this dataset."""
    class_names = get_class_names(dataset)
    num_classes = len(class_names)
    model = te.build_model("resnet50", num_classes=num_classes, img_size=img_size)
    _, ckpt_path = CHECKPOINTS[dataset]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if device is not None:
        model.to(device)
    if eval_mode:
        model.eval()
    return model, class_names


def build_eval_dataset(dataset: str, img_size: int = 224):
    """Returns a torchvision ImageFolder over <dataset>/test using the exact
    same (no-augmentation) eval transform train_eval.py's build_dataloaders
    uses for its test_loader, plus the matching mean/std for de-normalizing
    back to a displayable image."""
    from torchvision import datasets, transforms
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds = datasets.ImageFolder(DATASETS_DIR / dataset / "test", transform=eval_tf)
    return ds, mean, std
