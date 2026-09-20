> Historical research note from the source workspace. Paths, commands, and run results may refer to earlier experiments. See [the current training guide](../TRAINING.md) and [migration notes](../MIGRATION.md) before running a command.

# Controlled overfitting experiments

This package reuses the prepared dataset, two-head PatchTST, patch targets,
focal/Huber losses and normalization. It does not modify the original generator,
prepared recordings or prior checkpoints. The frozen plan and results live in
`reports/overfit_20260916`.

The completed results are in
[`reports/overfit_20260916/report.md`](../../reports/overfit_20260916/report.md).
Rotation helped roughness with a detection trade-off; stronger regularization
and smaller capacity did not jointly solve overfitting. Original defaults remain.

Nine configurations × three seeds compare uniform sampling, hierarchical
family/100-m-block/view sampling, a smaller encoder, stronger dropout/weight
decay, shared IMU rotation, small observation noise/bias, and their combination.
The combination is also compared with a real-only baseline. All runs use CUDA
BF16, batch 256 and LR 1e-4. Validation occurs every 256 updates, with a 12,288
update cap and the same early-stopping rule for every arm. Checkpoints minimize
the combined real-validation loss; disturbance threshold remains 0.5.

The block sampler preserves every baseline slot's real/synthetic and
Kaggle/LiRA profile stratum. Within a stratum it samples a family, spatial block,
recording/view and eligible start. LiRA uses reference section IDs, simulation
uses native station, and Kaggle uses integrated valid speed. Unknown centers
group by the nearest supported section/station inside that window, without
imputing targets or creating extra temporal groups. This changes sampling weights,
not the amount of independent road data or the stride-one dataset interface.

Augmentation happens on raw SI inputs only while training. One constant proper
rotation of at most five degrees is shared by acceleration, gravity and gyro.
Small noise is tied across identical held sensor vectors; constant per-window
biases preserve temporal structure. Missing gyro stays absent with its mask.
Speed, timestamps, labels and their masks are unchanged. The noise magnitudes
are declared conservative priors, not estimates from held-out recordings.

Shared frame rotation has a geometric basis in IMU augmentation;
[this primary sensor-learning paper](https://www.nature.com/articles/s41598-026-64846-5)
discusses that rationale. Its activity-recognition results do not establish
benefit for road sensing; the separate rotation/noise ablations test that here.
Time warping and arbitrary signal scaling are omitted because they can alter
the speed/response relationship and roughness interpretation.

```bash
# Write a new plan once; refuses to overwrite an existing experiment.
.venv/bin/python -m road_training.experiments.overfit.study --plan
.venv/bin/python -m road_training.experiments.overfit.run_suite
.venv/bin/python -m road_training.experiments.overfit.evaluate
.venv/bin/python -m road_training.experiments.overfit.report
```

The live `progress.json`, individual logs and per-run `history.json` contain
update-level progress. `report` also works on completed subsets. At each run's
best and last checkpoints a fixed clean TRAIN probe is evaluated with dropout
and augmentation disabled. Online augmented TRAIN scores should not be directly
treated as an evaluation-mode generalization gap.

These are development comparisons on the existing validation roads. No TEST
arrays are read. Multiple configurations and validation checks create selection
uncertainty; three seeds are not three independent roads. A fresh road/device
holdout remains necessary for a new final transfer claim.
