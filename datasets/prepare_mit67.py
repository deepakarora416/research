"""
Downloads MIT67 Indoor Scenes from Hugging Face and writes it to disk in
torchvision ImageFolder layout:

    datasets/mit67/train/<class>/*.jpg   (train + validation splits combined)
    datasets/mit67/test/<class>/*.jpg    (test split)

Usage:
    python prepare_mit67.py
"""

from pathlib import Path
from datasets import load_dataset, concatenate_datasets

OUT_DIR = Path(__file__).parent / "mit67"


def save_split(ds, split_name: str, class_names: list):
    split_dir = OUT_DIR / split_name
    for idx, example in enumerate(ds):
        label = class_names[example["labels"]]
        cls_dir = split_dir / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = example["image"].convert("RGB")
        img.save(cls_dir / f"{idx:06d}.jpg")


def main():
    print("Downloading keremberke/indoor-scene-classification (MIT67) from Hugging Face...")
    ds = load_dataset("keremberke/indoor-scene-classification", revision="refs/convert/parquet")

    class_names = ds["train"].features["labels"].names
    print(f"Classes ({len(class_names)}): {class_names}")

    train_pool = concatenate_datasets([ds["train"], ds["validation"]])
    save_split(train_pool, "train", class_names)
    print(f"Saved train split ({len(train_pool)} images)")

    save_split(ds["test"], "test", class_names)
    print(f"Saved test split ({len(ds['test'])} images)")

    print(f"\nDone. Data written to {OUT_DIR}")


if __name__ == "__main__":
    main()
