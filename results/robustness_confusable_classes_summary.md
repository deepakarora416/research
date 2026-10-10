# Robustness: confusable-class stress test (SUN397)

Addresses the PhD guide's explicit example directly: SUN397's class list
literally contains both `airport terminal` and `shopping mall indoor` as
separate classes (confirmed against `datasets/sun397/test/` folder names).
Full SUN397 test-set inference (3,970 images, 397 classes, 10 test images per
class) was run for both the CG-CBAT (`cgcbat_v6_frozen_sun397`) and plain
ResNet50 (`resnet50_8ep_sun397`) finalized checkpoints, a full 397x397
confusion matrix was built for each, and 4 visually-confusable class pairs
were extracted from it. Raw numbers: `results/robustness_confusable_classes.json`.

**Caveat up front:** SUN397 has only 10 test images per class, so every count
below is out of 10 -- a single misclassified image moves a percentage by 10
points. These are directionally informative, not statistically strong,
numbers.

## Sanity check

Overall SUN397 accuracy from this run: CG-CBAT 58.11%, ResNet50 58.26% (CG-CBAT
-0.15pp) -- matches the known existing TIE result exactly.

## Pair 1: airport terminal <-> shopping mall indoor (the PhD guide's example)

| | CG-CBAT | ResNet50 |
|---|---|---|
| airport terminal misclassified AS shopping mall indoor | 0/10 | 0/10 |
| shopping mall indoor misclassified AS airport terminal | 1/10 | 1/10 |
| airport terminal per-class accuracy | 3/10 (30%) | 3/10 (30%) |
| shopping mall indoor per-class accuracy | 3/10 (30%) | 5/10 (50%) |

Both models handle this specific confusion identically (one shopping-mall
image mistaken for an airport terminal, zero in the other direction). Neither
model shows the dramatic airport<->mall confusion the PhD guide's example
implies -- cross-confusion between THESE TWO classes specifically is rare.
That said, both models' overall per-class accuracy on `airport terminal` is
low (30% for both), so most of their errors on this class are being
misclassified as something else entirely, not specifically as `shopping mall
indoor`. On `shopping mall indoor`, ResNet50 is notably more accurate (50%
vs. 30% for CG-CBAT) but not because of cross-confusion with airport
terminal -- CG-CBAT's extra errors are going elsewhere.

## Pair 2: forest broadleaf <-> forest needleleaf

| | CG-CBAT | ResNet50 |
|---|---|---|
| broadleaf misclassified AS needleleaf | 0/10 | 2/10 |
| needleleaf misclassified AS broadleaf | 0/10 | 0/10 |

CG-CBAT has zero cross-confusion between these two visually-similar forest
subtypes; ResNet50 confuses 2/10 broadleaf images as needleleaf. A genuine,
if small-sample, point in CG-CBAT's favor.

## Pair 3: conference room <-> conference center

| | CG-CBAT | ResNet50 |
|---|---|---|
| room misclassified AS center | 1/10 | 1/10 |
| center misclassified AS room | 0/10 | 0/10 |

Identical for both models -- no difference.

## Pair 4: kitchen <-> kitchenette

| | CG-CBAT | ResNet50 |
|---|---|---|
| kitchen misclassified AS kitchenette | 5/10 (50%) | 3/10 (30%) |
| kitchenette misclassified AS kitchen | 0/10 | 3/10 (30%) |
| total cross-confusions (both directions) | 5/20 | 6/20 |

This is the most interesting pair and a mixed result. CG-CBAT confuses
`kitchen` as `kitchenette` much more often than ResNet50 does (50% vs. 30% of
kitchen images) -- notably worse in that specific direction. But CG-CBAT
never confuses `kitchenette` as `kitchen` (0/10), while ResNet50 does so 30%
of the time. Total cross-confusion count is similar (5 vs. 6 out of 20
relevant images), just distributed very differently: CG-CBAT is
asymmetrically biased toward predicting "kitchenette" whenever the scene is
small/kitchen-like, while ResNet50 confuses the pair more symmetrically in
both directions.

## Honest overall read

This stress test does **not** show CG-CBAT clearly outperforming ResNet50 at
telling confusable scene categories apart, consistent with SUN397 being the
dataset where CG-CBAT is an overall TIE/slight loss (-0.15pp) against
ResNet50. Results are pair-dependent and mixed:

- **airport terminal / shopping mall indoor** (the PhD guide's own example):
  no meaningful difference between the two models -- both handle this
  specific pair about equally well (and both struggle with `airport
  terminal` overall, for reasons unrelated to this particular confusion).
- **forest broadleaf / needleleaf**: CG-CBAT is cleaner (0 cross-confusion vs.
  ResNet50's 2/10).
- **conference room / center**: no difference.
- **kitchen / kitchenette**: roughly a wash in total error count, but CG-CBAT
  shows a stronger one-directional bias (kitchen -> kitchenette) than
  ResNet50 does.

**Recommendation for the thesis:** this is not strong evidence that CG-CBAT's
added attention/context-gating machinery specifically helps disambiguate
closely related scene categories. It is at best a narrow, pair-specific
advantage (forest subtypes) offset by a pair-specific new weakness (kitchen
bias), on top of a dataset where the two models already tie overall. Any
claim here should be stated as "mixed, pair-dependent, no clear overall
advantage" rather than a general robustness win, and the PhD guide's specific
airport/mall example should be reported as "both models handle this
confusion about equally (and both still struggle with `airport terminal`
generally)" rather than as a point of differentiation.
