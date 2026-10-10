# Pending Work

Tracked here so it survives across sessions. Update status as items are completed.

## For PhD thesis
- [ ] **Places365** — deprioritized early during architecture debugging, never finished.
      Need: ResNet50 baseline (8ep, matched) + CG-CBAT using the final, proven recipe
      (frozen backbone + validation-split early stopping) directly — no need to re-run
      the old v1-v5 architecture iteration, we already know the winning method.
- [ ] **Thesis write-up** — all experiments/results exist, but no thesis chapters or
      structured document have been drafted yet.

## For SCI Q1 paper (bigger lift)
- [ ] **Multi-seed validation** — every result so far is a single training run. Margins
      are small (0.00-2.12pp on 3 datasets, exact ties on 2). Need ~3 seeds per dataset
      for the final CG-CBAT model to show the result isn't just one lucky/unlucky run.
- [ ] **Clean, complete ablation table** — v1->v6 each added a component, but testing
      was scattered (early versions only tried on 2 of 5 fast datasets). Need one table:
      same 5 datasets, each component (differential LR, warm-up, multi-scale fusion,
      dropout/regularization, frozen-backbone strategy) toggled on/off systematically.
- [ ] **Broader baseline comparison** — currently only CG-CBAT vs. plain ResNet50 (plus
      a loosely-matched Swin-T reference, not trained to the same matched-epoch standard).
      Q1 reviewers will likely expect at least one more published attention method
      (e.g. plain CBAM-ResNet without the Transformer branch, or SE-Net) in the comparison.

## Completed (for reference)
- [x] Swin-T vs ResNet50 baseline across 6 datasets
- [x] CG-CBAT architecture (v1-v6) with iterative fixes
- [x] Frozen-backbone training strategy (the real fix for joint-training instability)
- [x] Final validated scoreboard: 3 wins, 2 ties, 0 losses (5 datasets, honest methodology)
- [x] Test-set leakage bug caught and fixed (proper validation split)
- [x] Interpretability analysis (Grad-CAM + CBAM attention maps)
- [x] Robustness analysis (corruption testing + confusable-class stress test)
