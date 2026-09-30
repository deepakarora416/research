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
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import timm


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


def train_one_epoch(model, loader, optimizer, criterion, device, log_every=20):
    model.train()
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
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    train_loader, test_loader, class_names = build_dataloaders(
        args.data_dir, args.batch_size, img_size=args.img_size, workers=args.workers
    )
    print(f"Classes: {class_names}")
    print(f"Train size: {len(train_loader.dataset)}, Test size: {len(test_loader.dataset)}")

    model = timm.create_model(args.model, pretrained=True, num_classes=len(class_names))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 1
    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from checkpoint at epoch {ckpt['epoch']} ({args.checkpoint})", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - time: {elapsed:.1f}s", flush=True)

        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, args.checkpoint)
        print(f"Saved checkpoint to {args.checkpoint} (epoch {epoch})", flush=True)

    results = evaluate(model, test_loader, class_names, device)
    print_table(results, class_names)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
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
        }, f, indent=2)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
