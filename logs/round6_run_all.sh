#!/bin/zsh
set -e
cd /Users/deepak/repo/research/experiments/baseline
source ../../venv/bin/activate

echo "=== scene15 MINIMAL frozen ===" 
python train_eval.py --data-dir ../../datasets/scene15 --model frozen_minimal_resnet50 \
  --epochs 8 --batch-size 32 --lr 1e-4 \
  --backbone-init-checkpoint ../../results/resnet50_8ep_scene15_checkpoint.pt \
  --out ../../results/frozen_minimal_scene15.json \
  --checkpoint ../../results/frozen_minimal_scene15_checkpoint.pt

echo "=== scene15 CG-CBAT v6 frozen ==="
python train_eval.py --data-dir ../../datasets/scene15 --model cg_cbat_resnet50 \
  --epochs 8 --batch-size 32 --lr 1e-4 \
  --backbone-init-checkpoint ../../results/resnet50_8ep_scene15_checkpoint.pt \
  --out ../../results/cgcbat_v6_frozen_scene15.json \
  --checkpoint ../../results/cgcbat_v6_frozen_scene15_checkpoint.pt

echo "=== sun397 MINIMAL frozen ==="
python train_eval.py --data-dir ../../datasets/sun397 --model frozen_minimal_resnet50 \
  --epochs 8 --batch-size 32 --lr 1e-4 \
  --backbone-init-checkpoint ../../results/resnet50_8ep_sun397_checkpoint.pt \
  --out ../../results/frozen_minimal_sun397.json \
  --checkpoint ../../results/frozen_minimal_sun397_checkpoint.pt

echo "=== sun397 CG-CBAT v6 frozen ==="
python train_eval.py --data-dir ../../datasets/sun397 --model cg_cbat_resnet50 \
  --epochs 8 --batch-size 32 --lr 1e-4 \
  --backbone-init-checkpoint ../../results/resnet50_8ep_sun397_checkpoint.pt \
  --out ../../results/cgcbat_v6_frozen_sun397.json \
  --checkpoint ../../results/cgcbat_v6_frozen_sun397_checkpoint.pt

echo "=== ALL DONE ==="
