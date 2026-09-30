"""
Downloads a stratified subset of SUN397 from Hugging Face and writes it to
disk in torchvision ImageFolder layout.

SUN397 has 397 classes and only ships a single ~100-images-per-class split
(no official train/test on this mirror), so we take a fixed number of images
per class for train and test (stratified, not random overlap) to keep this a
"small dataset" baseline comparable in scale to the others.

    datasets/sun397/train/<class>/*.jpg  (30 images/class)
    datasets/sun397/test/<class>/*.jpg   (10 images/class)

Usage:
    python prepare_sun397.py
"""

from collections import defaultdict
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "sun397"
TRAIN_PER_CLASS = 30
TEST_PER_CLASS = 10


def main():
    print("Streaming tanganke/sun397 from Hugging Face (stratified subset)...")
    ds = load_dataset("tanganke/sun397", split="train", streaming=True)
    class_names = ds.features["label"].names
    print(f"Classes ({len(class_names)})")

    counts = defaultdict(int)
    needed_per_class = TRAIN_PER_CLASS + TEST_PER_CLASS
    total_needed = needed_per_class * len(class_names)
    saved = 0

    for example in ds:
        label_idx = example["label"]
        label = class_names[label_idx]
        c = counts[label]
        if c >= needed_per_class:
            continue

        split_name = "train" if c < TRAIN_PER_CLASS else "test"
        cls_dir = OUT_DIR / split_name / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = example["image"].convert("RGB")
        img.save(cls_dir / f"{c:04d}.jpg")

        counts[label] += 1
        saved += 1
        if saved % 2000 == 0:
            print(f"  saved {saved}/{total_needed} images...", flush=True)

        if all(counts[c] >= needed_per_class for c in class_names):
            break

    print(f"Saved {saved} images total")
    print(f"\nDone. Data written to {OUT_DIR}")


if __name__ == "__main__":
    main()
