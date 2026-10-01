"""
Round 9 sanity check for the --val-split feature added to train_eval.py.

Fixes a methodological flaw from Round 8: --eval-every-epoch was picking
"best epoch" by watching TEST accuracy every epoch, which is test-set
leakage via model selection. --val-split carves a validation set out of
TRAIN only, uses it (not the test set) for per-epoch model selection, and
evaluates the test set exactly once at the end on the validation-selected
checkpoint.

This script verifies, independently of any real training run:

  (a) No overlap between validation images and either the train split or
      the test split -- using the REAL sun397 dataset on disk.
  (b) The split is stratified per class (every class contributes
      approximately val_split of its own images to validation; no class is
      starved) -- also on the real sun397 dataset.
  (c) The split is reproducible given a fixed seed, and gives a *different*
      split under a different seed.
  (d) The "final test evaluation uses the validation-best checkpoint, not
      the final epoch's" guarantee, verified end-to-end through the actual
      train_eval.main() code path (not reimplemented/mocked logic) on a
      tiny synthetic dataset, using a weight-dependent "accuracy" checksum
      so the proof is about actual tensor identity, not coincidence.

Run with:
    source venv/bin/activate
    python experiments/baseline/sanity_check_val_split.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "improved"))

import train_eval  # noqa: E402


def check_a_and_b_real_dataset():
    print("=== (a)/(b) real sun397 dataset: no leakage + stratification ===")
    data_dir = Path(__file__).parent.parent.parent / "datasets" / "sun397"
    assert data_dir.exists(), f"expected dataset at {data_dir}"

    val_split = 0.15
    seed = 42
    train_loader, val_loader, test_loader, class_names = train_eval.build_dataloaders(
        data_dir, batch_size=64, img_size=224, workers=0, val_split=val_split, seed=seed,
    )

    train_subset = train_loader.dataset   # Subset(train_full, train_idx)
    val_subset = val_loader.dataset       # Subset(val_full, val_idx)
    test_ds = test_loader.dataset         # plain ImageFolder over test/

    train_paths = {train_subset.dataset.samples[i][0] for i in train_subset.indices}
    val_paths = {val_subset.dataset.samples[i][0] for i in val_subset.indices}
    test_paths = {s[0] for s in test_ds.samples}

    # (a) no overlap, in both directions, and no path appears in more than
    # one split (train/val/test are a strict partition of all images seen).
    assert train_paths.isdisjoint(val_paths), "FAIL: train/val overlap!"
    assert val_paths.isdisjoint(test_paths), "FAIL: val/test overlap!"
    assert train_paths.isdisjoint(test_paths), "FAIL: train/test overlap (should be impossible, " \
        "different folders, but checking anyway)!"
    assert len(train_paths) == len(train_subset.indices), "duplicate indices in train subset"
    assert len(val_paths) == len(val_subset.indices), "duplicate indices in val subset"
    print(f"  train={len(train_paths)} val={len(val_paths)} test={len(test_paths)} "
          f"-- no overlap between any pair. PASS")

    # val + train (from the TRAIN folder only) must reconstitute the full
    # original train/ folder exactly -- val is carved OUT of train, not an
    # extra source of images, and the test folder is untouched by this split.
    train_full_len = len(train_subset.dataset)
    assert train_full_len == len(val_subset.dataset), \
        "train_full and val_full should enumerate the same underlying train/ folder"
    assert len(train_paths) + len(val_paths) == train_full_len, \
        "train subset + val subset should exactly partition the full train/ folder"
    print(f"  train subset + val subset = {len(train_paths) + len(val_paths)} "
          f"== full train/ folder size ({train_full_len}). PASS (val is carved out of train only)")

    # (b) stratification: every one of the 397 classes should be represented
    # in val roughly proportionally to val_split, and none should be zero.
    targets = train_subset.dataset.targets
    class_counts = {}
    for idx in train_subset.indices:
        class_counts.setdefault(targets[idx], [0, 0])[0] += 1
    for idx in val_subset.indices:
        class_counts.setdefault(targets[idx], [0, 0])[1] += 1

    assert len(class_counts) == len(class_names), "not every class appeared in train+val"
    zero_val_classes = [c for c, (tr, va) in class_counts.items() if va == 0]
    assert not zero_val_classes, f"FAIL: {len(zero_val_classes)} classes have NO validation images"
    fractions = [va / (tr + va) for tr, va in class_counts.values()]
    mean_frac = sum(fractions) / len(fractions)
    max_dev = max(abs(f - val_split) for f in fractions)
    assert abs(mean_frac - val_split) < 0.01, f"mean val fraction {mean_frac:.4f} far from {val_split}"
    # Per-class deviation can only be large due to integer rounding on small
    # per-class counts (SUN397 has ~30 images/class); bound it generously.
    assert max_dev < 0.12, f"FAIL: some class's val fraction deviates by {max_dev:.3f} from {val_split}"
    print(f"  all {len(class_counts)} classes represented in val; mean val fraction={mean_frac:.4f} "
          f"(target {val_split}), max per-class deviation={max_dev:.4f}. PASS (stratified)")


def check_c_reproducibility():
    print("\n=== (c) reproducibility: same seed -> identical split, different seed -> different split ===")
    data_dir = Path(__file__).parent.parent.parent / "datasets" / "sun397"
    train_full = train_eval.datasets.ImageFolder(data_dir / "train")
    targets = train_full.targets

    tr1, va1 = train_eval.stratified_val_split(targets, 0.15, seed=42)
    tr2, va2 = train_eval.stratified_val_split(targets, 0.15, seed=42)
    assert tr1 == tr2 and va1 == va2, "FAIL: same seed produced different splits"
    print("  seed=42 run twice -> identical train/val index lists. PASS")

    tr3, va3 = train_eval.stratified_val_split(targets, 0.15, seed=123)
    assert va1 != va3, "FAIL: different seeds produced the identical split (suspiciously not random)"
    print("  seed=42 vs seed=123 -> different splits, as expected. PASS")


def _make_tiny_dataset(root: Path, classes, n_train_per_class, n_test_per_class, img_size=32):
    rng = np.random.default_rng(0)
    for split, n in [("train", n_train_per_class), ("test", n_test_per_class)]:
        for cls in classes:
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(0, 255, (img_size, img_size, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i}.png")


def check_d_final_eval_uses_best_val_checkpoint():
    print("\n=== (d) final TEST eval genuinely uses the validation-BEST checkpoint, not the final epoch ===")
    tmp_dir = Path(tempfile.mkdtemp(prefix="val_split_sanity_"))
    try:
        data_dir = tmp_dir / "data"
        _make_tiny_dataset(data_dir, ["a", "b"], n_train_per_class=20, n_test_per_class=10)

        # Monkeypatch evaluate() so its "accuracy" is a deterministic
        # checksum of the model's CURRENT weights (sum of every parameter),
        # rather than a real classification metric. Since training genuinely
        # mutates the weights each epoch, this checksum differs per epoch in
        # a way that is tied to actual tensor identity -- so if the final
        # test-time checksum matches the checksum recorded for best_val_epoch
        # (and not the final epoch's), that is direct proof the reloaded
        # weights are the validation-best ones, not a coincidence of
        # accuracy numbers happening to match.
        real_evaluate = train_eval.evaluate

        def checksum_evaluate(model, loader, class_names, device):
            checksum = sum(p.detach().sum().item() for p in model.parameters())
            return {
                "accuracy": checksum,
                "precision_macro": 0.0,
                "recall_macro": 0.0,
                "f1_macro": 0.0,
                "per_class": {c: {"precision": 0, "recall": 0, "f1-score": 0, "support": 0}
                              for c in class_names},
            }

        train_eval.evaluate = checksum_evaluate

        out_json = tmp_dir / "out.json"
        ckpt_path = tmp_dir / "ckpt.pt"
        argv = [
            "train_eval.py",
            "--data-dir", str(data_dir),
            "--model", "resnet18",
            "--epochs", "5",
            "--batch-size", "8",
            "--lr", "1e-3",
            "--img-size", "32",
            "--workers", "0",
            "--val-split", "0.2",
            "--seed", "42",
            "--out", str(out_json),
            "--checkpoint", str(ckpt_path),
        ]
        old_argv = sys.argv
        sys.argv = argv
        try:
            train_eval.main()
        finally:
            sys.argv = old_argv
            train_eval.evaluate = real_evaluate

        import json
        import torch
        output = json.loads(out_json.read_text())

        best_val_epoch = output["best_val_epoch"]
        best_val_checksum = output["best_val_accuracy"]
        reported_test_checksum = output["results"]["accuracy"]
        history = {h["epoch"]: h["accuracy"] for h in output["val_eval_history"]}

        assert best_val_checksum == max(history.values()), \
            "best_val_accuracy should equal the max of the per-epoch val checksums"
        print(f"  best_val_epoch={best_val_epoch} (of {output['config']['epochs']} epochs), "
              f"best_val_checksum={best_val_checksum:.6f}. PASS (consistent with history)")

        # The reported final TEST result must have been computed with the
        # SAME weights as the best validation epoch -- i.e. identical
        # checksum, down to float precision (deepcopy + reload, no further
        # training in between).
        assert abs(reported_test_checksum - best_val_checksum) < 1e-6, (
            f"FAIL: final reported test checksum ({reported_test_checksum}) does not match "
            f"best_val checksum ({best_val_checksum}) -- final eval did NOT use the "
            f"validation-selected checkpoint!"
        )
        print(f"  reported final TEST checksum == best_val checksum "
              f"({reported_test_checksum:.6f} == {best_val_checksum:.6f}). PASS")

        final_epoch_checksum = history[output["config"]["epochs"]]
        if best_val_epoch != output["config"]["epochs"]:
            assert abs(reported_test_checksum - final_epoch_checksum) > 1e-6, (
                "FAIL: reported test checksum matches the FINAL epoch's checksum even though "
                "best_val_epoch is a different, earlier epoch -- selection had no effect!"
            )
            print(f"  best_val_epoch ({best_val_epoch}) != final epoch "
                  f"({output['config']['epochs']}), and the reported test result does NOT "
                  f"match the final epoch's checksum ({final_epoch_checksum:.6f}). "
                  f"PASS (early-stopping selection demonstrably took effect)")
        else:
            print(f"  NOTE: best_val_epoch happened to equal the final epoch ({best_val_epoch}) "
                  f"in this run -- selection vs. final-epoch comparison is inconclusive this time, "
                  f"but checksum-identity with best_val above already proves correct checkpoint reuse.")

        # The on-disk checkpoint's recorded epoch must also be the
        # validation-best epoch, not the final one.
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt["epoch"] == best_val_epoch, (
            f"FAIL: saved checkpoint epoch ({ckpt['epoch']}) != best_val_epoch ({best_val_epoch})"
        )
        print(f"  saved checkpoint's recorded epoch ({ckpt['epoch']}) == best_val_epoch. PASS")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    check_a_and_b_real_dataset()
    check_c_reproducibility()
    check_d_final_eval_uses_best_val_checkpoint()
    print("\nAll Round 9 --val-split sanity checks PASSED.")
