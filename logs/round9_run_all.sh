#!/bin/zsh
set -e
cd /Users/deepak/repo/research/experiments/baseline
source ../../venv/bin/activate

echo "=== SUN397 CG-CBAT v8 (frozen backbone, 16ep, val-split 0.15) ==="
python train_eval.py --data-dir ../../datasets/sun397 --model cg_cbat_resnet50 \
  --epochs 16 --batch-size 32 --lr 1e-4 \
  --backbone-init-checkpoint ../../results/resnet50_8ep_sun397_checkpoint.pt \
  --val-split 0.15 --seed 42 \
  --out ../../results/cgcbat_v8_val_sun397.json \
  --checkpoint ../../results/cgcbat_v8_val_sun397_checkpoint.pt

echo "=== Intel Image CG-CBAT v8 (frozen backbone, 8ep, val-split 0.15) ==="
python train_eval.py --data-dir ../../datasets/intel_image --model cg_cbat_resnet50 \
  --epochs 8 --batch-size 32 --lr 1e-4 \
  --backbone-init-checkpoint ../../results/resnet50_8ep_intel_image_checkpoint.pt \
  --val-split 0.15 --seed 42 \
  --out ../../results/cgcbat_v8_val_intel_image.json \
  --checkpoint ../../results/cgcbat_v8_val_intel_image_checkpoint.pt

echo "=== ALL DONE ==="
