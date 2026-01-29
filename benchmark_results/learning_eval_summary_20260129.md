# Learning Eval Summary (2026-01-29)

## Setup
- Dataset: CIFAR-10
- Projection: ConvProjection (256-dim) + MLPClassifier
- Subset: 20k, Val split: 0.1
- Label smoothing: 0.05, weight_decay: 1e-4
- Early stopping: 5

## Runs
- 0.0/0.0 (no constraints): best val_acc 0.2715
  - Report: benchmark_results/learning_eval_20260129_202615.json
- 0.25/0.25: best val_acc 0.2695
  - Report: benchmark_results/learning_eval_20260129_200336.json
- 0.5/0.5: best val_acc 0.108 (early stopped at epoch 7)
  - Report: benchmark_results/learning_eval_20260129_202207.json
- 0.25/0.25 extended (target 60 epochs): early stopped at epoch 32
  - Report: benchmark_results/learning_eval_20260129_203139.json

## Observations
- 0.25 nearly matches baseline at 20k/30 epochs; 0.5 collapses quickly.
- Extended 0.25 run improves until ~epoch 27, then plateaus; no late-phase jump.

## Interpretation
- Mild constraints (0.25) preserve learnability; strong constraints over-regularize.
- Learning curve suggests capacity or signal limits rather than under-training.