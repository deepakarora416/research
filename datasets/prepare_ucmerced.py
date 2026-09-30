"""
Downloads UC Merced Land Use from Hugging Face and writes a stratified
80/20 train/test split to disk in torchvision ImageFolder layout.

UC Merced ships only a single split (100 images/class, 21 classes), so we
split it ourselves: 80 train / 20 test per class.

    datasets/ucmerced/train/<class>/*.jpg
    datasets/ucmerced/test/<class>/*.jpg

Usage:
    python prepare_ucmerced.py
"""

from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "ucmerced"
TRAIN_PER_CLASS = 80
TEST_PER_CLASS = 20


def main():
    print("Downloading blanchon/UC_Merced from Hugging Face...")
    ds = load_dataset("blanchon/UC_Merced", split="train")
    class_names = ds.features["label"].names
    print(f"Classes ({len(class_names)}): {class_names}")

    counts = defaultdict(int)
    for example in ds:
        label = class_names[example["label"]]
        c = counts[label]
        split_name = "train" if c < TRAIN_PER_CLASS else "test"
        cls_dir = OUT_DIR / split_name / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = example["image"].convert("RGB")
        img.save(cls_dir / f"{c:04d}.jpg")
        counts[label] += 1

    print(f"Saved {sum(min(c, TRAIN_PER_CLASS) for c in counts.values())} train, "
          f"{sum(max(0, c - TRAIN_PER_CLASS) for c in counts.values())} test images")
    print(f"\nDone. Data written to {OUT_DIR}")


if __name__ == "__main__":
    main()
