"""
Organizes the official Places365 validation set (downloaded directly from
MIT CSAIL servers) into torchvision ImageFolder layout, using a fixed
stratified subset per class (consistent with the SUN397 subset approach).

Prerequisites (already downloaded to datasets/_places365_raw/):
    - val_256.tar                      (36,500 validation images, 256x256)
    - places365_val.txt                (filename -> class index)
    - categories_places365.txt         (class index -> class name)

    datasets/places365/train/<class>/*.jpg  (30 images/class)
    datasets/places365/test/<class>/*.jpg   (10 images/class)

Usage:
    python prepare_places365.py
"""

import shutil
from collections import defaultdict
from pathlib import Path

RAW_DIR = Path(__file__).parent / "_places365_raw"
OUT_DIR = Path(__file__).parent / "places365"
TRAIN_PER_CLASS = 30
TEST_PER_CLASS = 10


def main():
    categories = {}
    with open(RAW_DIR / "categories_places365.txt") as f:
        for line in f:
            path, idx = line.strip().split()
            # e.g. "/a/apartment_building/outdoor" -> "apartment_building_outdoor"
            # (first segment is just an alphabetic bucket, not part of the name)
            categories[int(idx)] = "_".join(path.strip("/").split("/")[1:])

    labels = {}
    with open(RAW_DIR / "places365_val.txt") as f:
        for line in f:
            filename, idx = line.strip().split()
            labels[filename] = int(idx)

    print(f"Classes: {len(categories)}, labeled images: {len(labels)}")

    per_class_files = defaultdict(list)
    for filename, idx in labels.items():
        per_class_files[idx].append(filename)

    needed_per_class = TRAIN_PER_CLASS + TEST_PER_CLASS
    saved = 0
    for idx, files in per_class_files.items():
        class_name = categories[idx]
        files = sorted(files)[:needed_per_class]
        for i, filename in enumerate(files):
            split_name = "train" if i < TRAIN_PER_CLASS else "test"
            cls_dir = OUT_DIR / split_name / class_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(RAW_DIR / "val_256" / filename, cls_dir / filename)
            saved += 1

    print(f"Saved {saved} images total")
    print(f"\nDone. Data written to {OUT_DIR}")


if __name__ == "__main__":
    main()
