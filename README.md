# Improved ML Model for Natural Scene Recognition

Thesis research project. Goal: establish a Swin-T baseline on 2-3 scene recognition
datasets, identify a concrete architectural gap, propose an improvement, and compare
results (precision, recall, F1, accuracy).

## Plan

1. **Baseline** — Fine-tune Swin-T on 2-3 scene-focused datasets (not generic object
   datasets). Candidates: MIT67 Indoor Scenes, SUN397, Places365 (subset).
   Report precision, recall, F1, accuracy per dataset.
2. **Gap analysis** — Swin-T's shifted-window attention is local-context biased;
   scene recognition depends more on global layout/context than object recognition
   does. This is the candidate weakness to target.
3. **Improvement** — Candidate directions (pick one, don't try all):
   - Hybrid local-global attention branch injected into Swin stages
   - Multi-scale feature pyramid fusion across Swin's hierarchical stages
   - Cross-dataset generalization / small-sample robustness improvements
   - Lightweight/efficient variant for edge deployment
4. **Re-evaluate** — Same datasets, same metrics, direct comparison table + discussion.
5. **Literature check** — Verify chosen gap isn't already closed by recent
   (2024-2026) papers before committing engineering time.

## Structure

- `literature/` — papers, notes, gap analysis
- `datasets/` — dataset notes, splits, preprocessing scripts
- `experiments/baseline/` — Swin-T baseline training/eval code
- `experiments/improved/` — improved model code
- `results/` — metrics, tables, plots for thesis
- `notes/` — working notes, decisions, meeting logs
