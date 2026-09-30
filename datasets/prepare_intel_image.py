"""
Downloads the Intel Image Classification dataset from Hugging Face and
writes it to disk in torchvision ImageFolder layout:

    datasets/intel_image/train/<class>/*.jpg
    datasets/intel_image/test/<class>/*.jpg

Usage:
    python prepare_intel_image.py
"""

from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path(__file__).parent / "intel_image"


def save_split(ds, split_name: str, class_names: list):
    split_dir = OUT_DIR / split_name
    for idx, example in enumerate(ds):
        label = class_names[example["label"]]
        cls_dir = split_dir / label
        cls_dir.mkdir(parents=True, exist_ok=True)
        img = example["image"].convert("RGB")
        img.save(cls_dir / f"{idx:06d}.jpg")


def main():
    print("Downloading sfarrukhm/intel-image-classification (parquet) from Hugging Face...")
    ds = load_dataset("sfarrukhm/intel-image-classification", revision="refs/convert/parquet")

    class_names = ds["train"].features["label"].names
    print(f"Classes: {class_names}")

    save_split(ds["train"], "train", class_names)
    print(f"Saved train split ({len(ds['train'])} images)")

    test_split = "test" if "test" in ds else "validation"
    save_split(ds[test_split], "test", class_names)
    print(f"Saved test split ({len(ds[test_split])} images)")

    print(f"\nDone. Data written to {OUT_DIR}")


if __name__ == "__main__":
    main()
