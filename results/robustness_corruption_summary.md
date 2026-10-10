# Robustness: Gaussian blur / noise corruption (CG-CBAT vs. ResNet50)

Full test-set evaluation (reusing `evaluate()` from `train_eval.py` unchanged)
on Intel Image (the TIE case) and MIT67 (the WIN case), at 7 conditions each:
clean, Gaussian blur (sigma 1.0 / 2.0 / 3.0, applied in raw pixel space before
normalization), Gaussian noise (std 0.1 / 0.25 / 0.5, added after
normalization). Raw numbers: `results/robustness_corruption.json`.

## Sanity check (passed)

| Dataset | Model | Our clean-severity accuracy | Known existing accuracy |
|---|---|---|---|
| Intel Image | CG-CBAT | 93.63% | 93.63% |
| Intel Image | ResNet50 | 93.63% | 93.63% |
| MIT67 | CG-CBAT | 76.00% | 76.00% |
| MIT67 | ResNet50 | 73.88% | 73.88% |

Exact match in all four cases -- the corrupted-condition numbers below can be
trusted as directly comparable to these baselines.

## Results

### Intel Image (accuracy %, and pp dropped from clean in parentheses)

| Condition | CG-CBAT | ResNet50 |
|---|---|---|
| clean | 93.63 (-) | 93.63 (-) |
| blur sigma=1.0 | 92.53 (-1.10) | 92.23 (-1.40) |
| blur sigma=2.0 | 86.80 (-6.83) | 84.47 (-9.17) |
| blur sigma=3.0 | 73.73 (-19.90) | 69.60 (-24.03) |
| noise std=0.1 | 91.70 (-1.93) | 92.03 (-1.60) |
| noise std=0.25 | 84.03 (-9.60) | 82.30 (-11.33) |
| noise std=0.5 | 57.93 (-35.70) | 55.40 (-38.23) |

**On Intel Image, CG-CBAT degrades more gracefully than ResNet50 in 5 of 6
corrupted conditions** -- it both stays higher in absolute accuracy AND drops
fewer percentage points from its own clean baseline at every blur severity
and at the two higher noise severities. The one exception is the mildest
noise level (std=0.1), where ResNet50 is marginally better (92.03% vs.
91.70%, i.e. CG-CBAT drops 1.93pp vs. ResNet50's 1.60pp) -- a small,
honestly-reported inversion, not hidden.

### MIT67 (accuracy %, and pp dropped from clean in parentheses)

| Condition | CG-CBAT | ResNet50 |
|---|---|---|
| clean | 76.00 (-) | 73.88 (-) |
| blur sigma=1.0 | 63.63 (-12.38) | 65.06 (-8.81) |
| blur sigma=2.0 | 30.75 (-45.25) | 35.88 (-38.00) |
| blur sigma=3.0 | 13.69 (-62.31) | 17.44 (-56.44) |
| noise std=0.1 | 69.25 (-6.75) | 69.69 (-4.19) |
| noise std=0.25 | 54.44 (-21.56) | 54.38 (-19.50) |
| noise std=0.5 | 18.56 (-57.44) | 17.37 (-56.50) |

**On MIT67, the result is the opposite of Intel Image, and not flattering for
CG-CBAT.** CG-CBAT's +2.12pp clean-accuracy lead over ResNet50 is completely
erased as soon as ANY corruption is applied:

- Under all three blur severities, CG-CBAT not only drops more pp than
  ResNet50, it falls to a LOWER absolute accuracy than ResNet50 (e.g. at
  sigma=2.0: CG-CBAT 30.75% vs. ResNet50 35.88% -- a 5.13pp deficit, having
  started 2.12pp AHEAD on clean images).
- Under noise, the picture is close to a wash: CG-CBAT is very slightly ahead
  only at the extreme std=0.5 (18.56% vs 17.37%, within noise of each other
  given this is a 1600-image test set), and behind or tied at std=0.1/0.25.
- Across every single one of the 6 corrupted MIT67 conditions, CG-CBAT's
  percentage-point drop from its own clean baseline is larger than
  ResNet50's drop from its clean baseline. By this "graceful degradation"
  metric, CG-CBAT is **less** robust than plain ResNet50 on MIT67, not more.

## Honest overall read

Robustness to these corruptions is **not a uniform win for CG-CBAT** -- it is
dataset- and corruption-type-dependent, and on one of our two test datasets
(MIT67, the one where CG-CBAT's clean accuracy was his best WIN case) it is
measurably worse under corruption than the plain backbone it's being
compared to, badly enough to flip from ahead to behind in absolute accuracy.
On Intel Image, CG-CBAT does degrade more gracefully under both blur and
heavier noise, so the story is genuinely mixed rather than a clean win or
clean loss.

A plausible (speculative, not verified further here) explanation consistent
with the architecture: CG-CBAT's FPN-style lateral fusion path explicitly
injects fine-grained mid-stage spatial/textural detail into the final feature
map (see model.py's v4 design notes) specifically to help on datasets with
more fine texture structure -- which is exactly the kind of signal Gaussian
blur destroys first. MIT67's indoor scenes (bookshelves, pipes, tiling,
furniture edges) may lean on that fine detail more than Intel Image's more
coarse-grained landscape categories do, which would explain why blur hurts
CG-CBAT disproportionately on MIT67 specifically but not on Intel Image. This
is offered as a hypothesis for why the effect might be asymmetric, not as a
proven mechanism -- it was not independently tested here (e.g. by ablating
the lateral path), and it should be presented as a hypothesis, not a finding,
if cited in the writeup.

**Recommendation for the thesis:** do not claim corruption robustness as a
general property of CG-CBAT. If robustness is discussed, report both
datasets' results as shown above -- the Intel Image result is a genuine
positive data point, but the MIT67 result is a genuine negative one, and
presenting only one would misrepresent the evidence.
