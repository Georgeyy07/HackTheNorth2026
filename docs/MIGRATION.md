# Source handoff — 2026-09-19

This branch contains the reusable `road_training` package, its tests and research drivers, and the `road_viewer` replay app. The source workspace and its running viewer/training artifacts were preserved.

## Organization

Reusable model, data, loss, augmentation, checkpoint, and streaming modules remain under `road_training`. Experiment orchestration moved to `road_training/experiments/`; conversion utilities remain under `tools/`. Historical Markdown notes moved to `docs/research/`. Small shared JSON, hashing, metrics, sampling, and evaluation helpers were brought into this package, removing imports from the old simulator repository.

`migration.json` maps the 128 copied source/documentation files to their new paths and records their original content hashes. Imports and source-provenance paths were updated for the new layout. Old experiment plans intentionally fail source-hash checks after migration; initialize fresh plans instead of editing their frozen receipts to bypass validation.

Portability changes include:

- An installable Python package with explicit optional dependencies and a repository-local test configuration.
- An ensemble loader independent of research runners, accepting both legacy absolute and receipt-relative checkpoint paths with checksum validation.
- A receipt-generation command and the existing selected timeline configuration.
- A viewer application factory and explicit export/profile directory arguments or environment variables; importing the module no longer requires local replay data.
- Self-contained API and streaming-boundary test fixtures; no private corpus needed for unit tests.
- Git exclusions for data, weights, generated artifacts, environments, browser binaries, and credentials.

## Validation

| Check | Result |
| --- | --- |
| Python unit/integration suite | 192 passed, including CUDA tests on this machine |
| JavaScript replay-state suite | 5 passed |
| Browser integration using the existing export | All 9 sessions, 509,109 samples; zero browser JavaScript errors |
| Existing four-checkpoint ensemble parity | Bit-identical original/transferred outputs in CPU FP32 and CUDA BF16 on a two-window fixture, including missing gyros |
| Module imports | All 77 non-test training modules imported successfully |
| Public CLI help | 8 training/pretraining/export/checkpoint/viewer entry points passed |
| Current training recipe initialization | New parent/mounting plans initialized and source/data hashes verified against the aligned corpus |
| Packaging | Python wheel built successfully |

Browser checks covered prediction availability, provisional/final updates, event timing, play/pause, speed, seeking, complete session coverage, GPS gaps, absent LiRA gyros, mobile layout, and unavailable map tiles. The new checkout was tested on a separate temporary port; the original viewer remained running.

These checks validate the transfer and software behavior. They do not constitute new model training, new dataset evaluation, or new real-world generalization evidence. Data and trained weights must be supplied separately. The physics simulator itself is outside this handoff; its prepared output remains supported by `RoadDataset`.
