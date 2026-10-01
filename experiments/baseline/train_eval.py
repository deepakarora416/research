"""
Model-agnostic baseline trainer/evaluator for natural scene recognition.

Fine-tunes any pretrained timm model (e.g. swin_tiny_patch4_window7_224,
resnet50) on a small scene dataset and reports per-class and overall
Precision, Recall, F1, and Accuracy. Full training config (batch size,
optimizer, loss function, learning rate, epochs, image size) is recorded in
the output JSON for reproducibility.

Usage:
    python train_eval.py --data-dir ../../datasets/intel_image \
        --model swin_tiny_patch4_window7_224 --epochs 3 --batch-size 32 \
        --out ../../results/swin_intel_image.json \
        --checkpoint ../../results/swin_intel_image_checkpoint.pt

    python train_eval.py --data-dir ../../datasets/intel_image \
        --model resnet50 --epochs 3 --batch-size 32 \
        --out ../../results/resnet50_intel_image.json \
        --checkpoint ../../results/resnet50_intel_image_checkpoint.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import timm

sys.path.insert(0, str(Path(__file__).parent.parent / "improved"))

CUSTOM_MODEL_NAME = "cg_cbat_resnet50"
MINIMAL_MODEL_NAME = "frozen_minimal_resnet50"  # Round 6 sanity-check floor
WARMUP_EPOCHS = 2  # cg_cbat_resnet50 only, and only when NOT using --backbone-init-checkpoint


def build_model(model_name: str, num_classes: int, img_size: int):
    """Returns a timm model, or one of our custom models when model_name
    matches CUSTOM_MODEL_NAME / MINIMAL_MODEL_NAME."""
    if model_name == CUSTOM_MODEL_NAME:
        from model import ContextGatedCBATNet
        return ContextGatedCBATNet(num_classes=num_classes, img_size=img_size)
    if model_name == MINIMAL_MODEL_NAME:
        from model import FrozenBackboneMinimalNet
        return FrozenBackboneMinimalNet(num_classes=num_classes, img_size=img_size)
    return timm.create_model(model_name, pretrained=True, num_classes=num_classes)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_dataloaders(data_dir: Path, batch_size: int, img_size: int = 224, workers: int = 4):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, test_loader, train_ds.classes


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=20,
                     freeze_backbone_bn=False):
    model.train()
    if freeze_backbone_bn:
        # model.train() above just re-enabled BatchNorm training-mode
        # (running-stats updates + batch statistics) on EVERY submodule,
        # including model.backbone. Round 6 freezes the backbone completely,
        # so its BN running stats must stay fixed at their trained values
        # too -- put it back in eval() after the blanket model.train() call.
        model.backbone.eval()
    total_loss = 0.0
    num_batches = len(loader)
    start = time.time()
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        if batch_idx % log_every == 0 or batch_idx == num_batches:
            elapsed = time.time() - start
            print(f"  batch {batch_idx}/{num_batches} - loss: {loss.item():.4f} - elapsed: {elapsed:.1f}s", flush=True)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, class_names, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    per_class = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "per_class": per_class,
    }


def print_table(results: dict, class_names: list):
    print("\n=== Per-class results ===")
    header = f"{'Class':<15}{'Precision':>10}{'Recall':>10}{'F1':>10}"
    print(header)
    print("-" * len(header))
    for cls in class_names:
        m = results["per_class"][cls]
        print(f"{cls:<15}{m['precision']:>10.3f}{m['recall']:>10.3f}{m['f1-score']:>10.3f}")

    print("\n=== Overall ===")
    print(f"{'Accuracy':<15}{results['accuracy']:.3f}")
    print(f"{'Precision (macro)':<20}{results['precision_macro']:.3f}")
    print(f"{'Recall (macro)':<20}{results['recall_macro']:.3f}")
    print(f"{'F1 (macro)':<20}{results['f1_macro']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory with train/ and test/ subfolders (ImageFolder format)")
    parser.add_argument("--model", default="swin_tiny_patch4_window7_224")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("../../results/baseline_results.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("../../results/baseline_checkpoint.pt"),
                        help="Path to save/resume training checkpoint after each epoch")
    parser.add_argument("--backbone-init-checkpoint", type=Path, default=None,
                        help="Round 6: path to a plain timm resnet50 train_eval.py checkpoint "
                             "(dict with 'model_state', full classifier-head model) whose backbone "
                             "conv/bn weights are loaded into the model's `backbone` submodule "
                             "before training. Only meaningful for --model cg_cbat_resnet50 or "
                             f"--model {MINIMAL_MODEL_NAME}. When set, the backbone is FROZEN for "
                             "the entire run (all epochs) instead of the usual warm-up-then-unfreeze "
                             "schedule -- this is the Round 6 frozen-backbone training strategy.")
    parser.add_argument("--eval-every-epoch", action="store_true",
                        help="Round 8: evaluate on the test set after EVERY epoch (not just the "
                             "final one), printing each epoch's accuracy/precision/recall/F1 to "
                             "stdout and reporting the best epoch's accuracy at the end (also "
                             "recorded in the output JSON as 'epoch_eval_history'/'best_epoch'/"
                             "'best_epoch_accuracy'). Works for any model. Off by default since "
                             "it adds wall-clock time; the normal final-epoch evaluation/JSON "
                             "output is unchanged either way.")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size, img_size=args.img_size, workers=args.workers
    )
    print(f"Classes: {class_names}")
    print(f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")

    model = build_model(args.model, num_classes=len(class_names), img_size=args.img_size)

    frozen_backbone_mode = args.backbone_init_checkpoint is not None
    if frozen_backbone_mode:
        if args.model not in (CUSTOM_MODEL_NAME, MINIMAL_MODEL_NAME):
            raise ValueError(
                f"--backbone-init-checkpoint is only supported for --model "
                f"{CUSTOM_MODEL_NAME} or {MINIMAL_MODEL_NAME}, got --model {args.model}"
            )
        # Round 6: load the fully-trained plain-ResNet50 checkpoint's backbone
        # weights, then freeze the backbone for the ENTIRE run. See
        # model.py's load_backbone_checkpoint() docstring for exactly how the
        # state-dict keys were verified to line up (identical key names/
        # shapes except the checkpoint's fc.weight/fc.bias, which features_only
        # has no use for and which we intentionally drop).
        num_matched, num_dropped = model.load_backbone_checkpoint(str(args.backbone_init_checkpoint))
        model.freeze_backbone()
        print(f"Round 6 frozen-backbone mode: loaded {num_matched} backbone tensors from "
              f"{args.backbone_init_checkpoint}, dropped {num_dropped} non-backbone keys, "
              f"backbone frozen for all {args.epochs} epochs.", flush=True)

    model.to(device)

    if args.model == CUSTOM_MODEL_NAME and not frozen_backbone_mode:
        # Differential LR: pretrained backbone needs small updates, new
        # randomly-initialized modules (CBAM/Transformer/gate) need to learn
        # much faster since they start from scratch.
        backbone_params = list(model.backbone.parameters())
        new_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        # v5 regularization fix: matched-epoch validation showed v4 (10x LR,
        # no weight decay on the new modules) overfitting badly in
        # low-images-per-class regimes (worst on SUN397). Reduced the new-
        # modules LR multiplier 10x -> 5x and added meaningful weight decay
        # to that group specifically; the backbone keeps its light 0.1x LR
        # and near-zero weight decay since it's pretrained, not random-init.
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.lr * 0.1, "weight_decay": 0.0},
            {"params": new_params, "lr": args.lr * 5, "weight_decay": 0.05},
        ])
        print(f"Using differential LR: backbone={args.lr * 0.1} (wd=0.0), "
              f"new modules={args.lr * 5} (wd=0.05)", flush=True)
    elif frozen_backbone_mode:
        # Round 6: backbone is permanently frozen (requires_grad=False), so
        # there's no LR-multiplier hack needed to protect it from noisy
        # gradients -- there simply are no gradients to it. Train the new
        # modules (or, for the MINIMAL model, just the classifier) at a
        # single standard LR, keeping the Round 5 weight decay as a
        # reasonable starting regularizer.
        new_params = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
        optimizer = torch.optim.AdamW([
            {"params": new_params, "lr": args.lr * 3, "weight_decay": 0.05},
        ])
        print(f"Round 6 frozen-backbone optimizer: new/head modules lr={args.lr * 3} (wd=0.05), "
              f"backbone excluded from optimizer updates (frozen).", flush=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']} ({args.checkpoint})", flush=True)

    epoch_eval_history = []
    best_epoch = None
    best_epoch_accuracy = None

    for epoch in range(start_epoch, args.epochs + 1):
        if args.model == CUSTOM_MODEL_NAME and not frozen_backbone_mode:
            # Warm-up: freeze the pretrained backbone for the first
            # WARMUP_EPOCHS epochs so the randomly-initialized new modules
            # (CBAM, Transformer branch, gate MLP) stabilize first, instead
            # of also nudging the backbone's good pretrained weights with
            # noisy gradients coming from those still-random modules. The
            # optimizer already has both param groups registered (with their
            # differential LRs) from the start -- setting requires_grad is
            # enough, since AdamW simply skips params whose .grad is None.
            backbone_frozen = epoch <= WARMUP_EPOCHS
            for p in model.backbone.parameters():
                p.requires_grad = not backbone_frozen
            if epoch == start_epoch or epoch == WARMUP_EPOCHS + 1:
                state = f"frozen (warm-up, {WARMUP_EPOCHS} epochs total)" if backbone_frozen else "unfrozen"
                print(f"Epoch {epoch}: backbone {state}", flush=True)
        elif frozen_backbone_mode and epoch == start_epoch:
            print(f"Epoch {epoch}: backbone permanently frozen (Round 6 mode, "
                  f"no warm-up schedule)", flush=True)

        start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            freeze_backbone_bn=frozen_backbone_mode,
        )
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - time: {elapsed:.1f}s", flush=True)

        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint} (epoch {epoch})", flush=True)

        if args.eval_every_epoch:
            epoch_results = evaluate(model, test_loader, class_names, device)
            print(
                f"Epoch {epoch} eval: accuracy={epoch_results['accuracy']:.4f} "
                f"precision_macro={epoch_results['precision_macro']:.4f} "
                f"recall_macro={epoch_results['recall_macro']:.4f} "
                f"f1_macro={epoch_results['f1_macro']:.4f}",
                flush=True,
            )
            epoch_eval_history.append({
                "epoch": epoch,
                "accuracy": epoch_results["accuracy"],
                "precision_macro": epoch_results["precision_macro"],
                "recall_macro": epoch_results["recall_macro"],
                "f1_macro": epoch_results["f1_macro"],
            })
            if best_epoch_accuracy is None or epoch_results["accuracy"] > best_epoch_accuracy:
                best_epoch_accuracy = epoch_results["accuracy"]
                best_epoch = epoch

    if args.eval_every_epoch and best_epoch is not None:
        print(
            f"\nBest epoch by test accuracy: epoch {best_epoch} "
            f"with accuracy={best_epoch_accuracy:.4f}",
            flush=True,
        )

    results = evaluate(model, test_loader, class_names, device)
    print_table(results, class_names)

    output = {
        "model": args.model,
        "dataset_dir": str(args.data_dir),
        "num_classes": len(class_names),
        "train_size": len(train_loader.dataset),
        "test_size": len(test_loader.dataset),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "optimizer": "AdamW",
            "loss_function": "CrossEntropyLoss",
            "image_size": args.img_size,
            "pretrained": True,
            "device": str(device),
        },
        "classes": class_names,
        "results": results,
    }
    if args.eval_every_epoch:
        # Round 8: additive fields only -- the keys/structure above are
        # unchanged so existing downstream parsing of the default (no-flag)
        # output format keeps working.
        output["eval_every_epoch"] = True
        output["epoch_eval_history"] = epoch_eval_history
        output["best_epoch"] = best_epoch
        output["best_epoch_accuracy"] = best_epoch_accuracy

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
