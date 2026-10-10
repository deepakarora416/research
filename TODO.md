# Pending Work

Tracked here so it survives across sessions. Update status as items are completed.

## For PhD thesis
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
- [x] Test-set leakage bug caught and fixed (proper validation split)
- [x] Interpretability analysis (Grad-CAM + CBAM attention maps)
- [x] Robustness analysis (corruption testing + confusable-class stress test)
- [x] Places365 finished with the proven recipe (frozen backbone + validation split)
- [x] **FINAL validated scoreboard across all 6 datasets, honest methodology, no test-set
      leakage: 3 wins (Scene15 +0.81pp, MIT67 +2.12pp, UC Merced +1.19pp), 2 ties
      (Intel Image 0.00pp, SUN397 -0.15pp), 1 loss (Places365 -2.19pp). Pattern: wins on
      moderate-data/moderate-difficulty tasks, ties on easy/near-ceiling or very-hard-
      but-not-impossible tasks, loses only on the most extreme data-scarcity regime
      (Places365: ~30 images/class across 365 classes, same regime as SUN397's tie).**
