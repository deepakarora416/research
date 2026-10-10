# Interpretability: CBAM spatial attention vs. Grad-CAM (CG-CBAT)

8 correctly-classified test images (4 UC Merced, 4 MIT67), each visualized two
ways using the finalized CG-CBAT checkpoints (`cgcbat_v6_frozen_ucmerced`,
`cgcbat_v6_frozen_mit67`):

1. **CBAM** -- the genuine 2D spatial attention map from
   `model.cbam_final.spatial_att`, captured via a forward hook (7x7, bilinear-
   upsampled to 224x224, min-max normalized per image for display).
2. **Grad-CAM** -- gradient of the predicted-class logit w.r.t. `gated_feat`
   (the final fused+gated 2048x7x7 feature map, i.e. the layer immediately
   before global average pooling), channel-averaged into weights, weighted-
   summed, ReLU'd, upsampled, normalized. Standard Grad-CAM, implemented with
   `torch.autograd.grad` (no library dependency, no model changes).

Files: `results/interpretability/<dataset>_<class>_0_{original,cbam,gradcam}.png`,
raw stats in `results/interpretability/stats.json`.

## Headline finding (be honest about it)

**The two methods tell different, partly contradictory stories.** Grad-CAM
localizes on genuinely scene-defining content in most examples. CBAM's final-
stage spatial attention map, by contrast, shows a **strong, recurring
positional bias toward the upper portion of the image** (or an upper-right
diagonal band) in essentially every single example, largely independent of
where the semantically relevant content actually is. This looks less like
"the model is attending to meaningful regions" and more like a learned
positional prior in that particular attention head. This is a real caveat to
flag, not a flattering result to hide.

## Per-example observations

### UC Merced

- **parkinglot** (conf 0.999): Grad-CAM lands squarely on the dense cluster of
  parked cars in the frame's center -- relevant. CBAM instead highlights a
  diagonal band running from the top edge down through the upper-right,
  roughly following one row of cars but not the lot as a whole -- partially
  relevant at best, looks more like an edge/diagonal-texture detector than a
  "this is a parking lot" detector.
- **golfcourse** (conf 0.648, the least confident example in the set): Both
  maps are weak here. Grad-CAM's hottest region sits on the tree line at the
  image's left/upper border, not on the distinctive mowed-stripe fairway
  pattern that's the actual giveaway for "golf course." CBAM again lights up
  the top-right corner, on trees. Neither map points convincingly at the
  fairway. This is the weakest example in the set and the low confidence is
  consistent with that.
- **forest** (conf 0.999): Image is homogeneous canopy texture edge-to-edge,
  so there's no single discriminative sub-region to find. Grad-CAM produces a
  generic centered blob (plausible under a center-bias prior for a texture-
  uniform scene, but not very informative). CBAM again concentrates near the
  top.
- **denseresidential** (conf 0.994): Grad-CAM covers the cluster of rooftops
  well -- genuinely the defining content for this class. CBAM again shows the
  same upper-right diagonal band seen in "parkinglot," on roof edges this
  time, consistent with the positional-bias read rather than a content match.

### MIT67

- **bookstore** (conf 0.990): Grad-CAM's single clearest hit in the set -- it
  lands tightly on the stacked book display table in the foreground, exactly
  the kind of shelves/books evidence a human would use. CBAM instead
  highlights the ceiling and fluorescent light fixture at the top of the
  frame -- irrelevant to "bookstore" as a concept.
- **kitchen** (conf 1.000): Grad-CAM centers on the range hood / cooktop
  island -- a strong, relevant choice. CBAM again sits on the ceiling area
  above the hood.
- **corridor** (conf 0.966): Grad-CAM highlights the vanishing point down the
  hallway -- the converging perspective lines are a genuinely distinctive
  "corridor" cue. CBAM lights up the ceiling/pipes in the upper-left corner.
- **airport_inside** (conf 1.000): Grad-CAM's hottest region is the empty
  floor tiles in front of the door/seating, not the seating or door
  themselves -- a middling result; plausible (open floor area is part of the
  "waiting area" gestalt) but not as sharply on-target as bookstore/kitchen/
  corridor. CBAM again shows the top-left corner band.

## Reading across all 8 (not cherry-picked)

- **Grad-CAM: 5/8 clearly on scene-defining content** (parkinglot, kitchen,
  corridor, bookstore, denseresidential), **2/8 middling** (forest -- diffuse/
  generic center bias on a texture-uniform scene; airport_inside -- floor
  rather than the furniture/door), **1/8 weak** (golfcourse -- misses the
  fairway, lands on tree border, also the lowest-confidence prediction).
- **CBAM: 0/8 land on the specific object/region a human would call
  "discriminative."** All 8 show high activation concentrated in the upper
  portion of the frame (ceiling, sky/treeline, or an upper-right diagonal
  band), regardless of whether the semantically relevant content is up there
  or not. The raw (pre-normalization) sigmoid outputs do have real dynamic
  range per image (typically ~0.5-0.8 at the low end up to ~0.9999 at the
  peak -- see `stats.json`), so it isn't a degenerate flat/no-op map; it is
  discriminating *something* spatially, but that something tracks image
  position far more than it tracks scene semantics in this sample.

## Honest takeaway for the thesis

Grad-CAM on the fused/gated feature map gives a reasonably convincing
qualitative story that the classifier's actual decision-relevant evidence
(books, cooktops, hallway geometry, car clusters) is concentrated in sensible
places most of the time, with real misses (golfcourse) that should be
disclosed rather than dropped from the writeup. The CBAM spatial-attention
module's own attention map, however, is **not** a reliable visualization of
"what the model is looking at" in this sample -- it looks positionally
biased rather than content-driven, and should not be presented alone as
evidence of interpretable attention. If the thesis wants an attention-based
interpretability claim, Grad-CAM on `gated_feat` is the stronger piece of
evidence of the two; CBAM's raw spatial map, as visualized here, undercuts
rather than supports an "attends to scene-relevant regions" narrative, and
that caveat should be stated explicitly if both are shown.
