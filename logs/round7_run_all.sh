#!/bin/zsh
set -e
cd /Users/deepak/repo/research/experiments/baseline
source ../../venv/bin/activate

for ds in intel_image mit67 ucmerced; do
  echo "=== $ds MINIMAL frozen ==="
  python train_eval.py --data-dir ../../datasets/$ds --model frozen_minimal_resnet50 \
    --epochs 8 --batch-size 32 --lr 1e-4 \
    --backbone-init-checkpoint ../../results/resnet50_8ep_${ds}_checkpoint.pt \
    --out ../../results/frozen_minimal_${ds}.json \
    --checkpoint ../../results/frozen_minimal_${ds}_checkpoint.pt

  echo "=== $ds CG-CBAT v6 frozen ==="
  python train_eval.py --data-dir ../../datasets/$ds --model cg_cbat_resnet50 \
    --epochs 8 --batch-size 32 --lr 1e-4 \
    --backbone-init-checkpoint ../../results/resnet50_8ep_${ds}_checkpoint.pt \
    --out ../../results/cgcbat_v6_frozen_${ds}.json \
    --checkpoint ../../results/cgcbat_v6_frozen_${ds}_checkpoint.pt
done

echo "=== ALL DONE ==="
