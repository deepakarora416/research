"""
Organizes the 15-Scene Image Dataset (Oliva & Torralba / Fei-Fei & Perona /
Lazebnik et al.) into torchvision ImageFolder layout.

The extracted archive only has numbered folders (00-14) with no class
names. We map them to class names by matching each folder's image count to
the published per-category counts from Lazebnik et al. (2006) "Beyond Bags
of Features" — all 15 counts are unique, so the mapping is unambiguous
(verified: they sum to exactly 4,485, the dataset's known total size).

Follows the dataset's standard evaluation protocol: 100 training images per
category (fixed, first 100 by filename), remaining images used for testing.

    datasets/scene15/train/<class>/*.jpg  (100 images/class)
    datasets/scene15/test/<class>/*.jpg   (remaining images/class)

Usage:
    python prepare_scene15.py
"""

import shutil
from pathlib import Path

RAW_DIR = Path(__file__).parent / "_scene15_raw" / "15-Scene"
OUT_DIR = Path(__file__).parent / "scene15"
TRAIN_PER_CLASS = 100

# folder -> (class name, published image count, used to verify the mapping)
FOLDER_TO_CLASS = {
    "00": ("bedroom", 216),
    "01": ("suburb", 241),
    "02": ("industrial", 311),
    "03": ("kitchen", 210),
    "04": ("livingroom", 289),
    "05": ("coast", 360),
    "06": ("forest", 328),
    "07": ("highway", 260),
    "08": ("insidecity", 308),
    "09": ("mountain", 374),
    "10": ("opencountry", 410),
    "11": ("street", 292),
    "12": ("tallbuilding", 356),
    "13": ("office", 215),
    "14": ("store", 315),
}


def main():
    total_saved = 0
    for folder, (class_name, expected_count) in FOLDER_TO_CLASS.items():
        src_dir = RAW_DIR / folder
        files = sorted(src_dir.iterdir())
        assert len(files) == expected_count, (
            f"Folder {folder} has {len(files)} images, expected {expected_count} "
            f"for class '{class_name}' — mapping may be wrong"
        )

        for i, src_file in enumerate(files):
            split_name = "train" if i < TRAIN_PER_CLASS else "test"
            cls_dir = OUT_DIR / split_name / class_name
            cls_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_file, cls_dir / src_file.name)
            total_saved += 1

    print(f"Saved {total_saved} images total across {len(FOLDER_TO_CLASS)} classes")
    print(f"\nDone. Data written to {OUT_DIR}")


if __name__ == "__main__":
    main()
