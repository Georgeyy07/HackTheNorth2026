# Data contract

`RoadDataset(root, source='real'|'synthetic'|'both', split='train'|'val'|'test')` reads a prepared corpus. It never downloads or regenerates data during training. `real_dataset='all'|'kaggle'|'lira'|'roadsens'` optionally filters the real component.

## Signals and windows

- `x`: float32 `[time, 7]`, ordered `accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, speed`.
- `mask`: boolean of the same shape. Missing readings use zero placeholders and false masks. Never interpret a missing gyro as a measured zero.
- Sampling: 100 Hz; accelerometer m/s² including gravity; gyro rad/s; speed m/s.
- Intended nominal vehicle-aligned frame: x forward, y left, z up. Source mount calibration is approximate; see [mounting evidence](research/MOUNTING.md). Phone measurements need the same frame before inference. The current mounting augmentation covers modest mounting offsets, not an arbitrary upside-down phone.
- A typical input is 1,024 samples (10.24 s), divided into 64 nonoverlapping 16-sample patches. Channels receive shared per-channel encoding; the heads combine the channel features.
- `InstancePatchTST` normalizes observed values separately for each window/channel. Window-scale descriptors can preserve amplitude information for the heads. Do not globally normalize its inputs first.

Each dataset item also contains time, recording ID, source, dataset name, and start index. With `return_labels=True`, per-sample label dictionaries are included; `train_multitask.patch_targets` creates the two patch-level targets and their validity masks.

## Prepared files

```text
manifest.json
records/<record-id>/
  x.npy                 [T,7] float32
  mask.npy              [T,7] bool
  time.npy              [T] float64
  labels.npy            [T,5] int16, unknown=-100
  iri.npy               [T] float32, unknown=NaN
  overall_iri.npy        [T] float32, optional section supervision
  roughness_section.npy [T] integer, paired with overall_iri.npy
  metadata.json
```

The manifest declares sample rate, ordered channels, records, `label_classes`, and TRAIN-only normalization statistics. Statistics remain part of the shared file schema even when the selected model performs instance normalization instead. Records declare `id`, relative `path`, sample count, source, split, and dataset provenance. Importers also store file checksums.

`labels.npy` columns are explicit defect, assumed-normal defect, five-class Kaggle type, synthetic type, and synthetic quality grade. These taxonomies are not interchangeable. The current two-head model uses binary disturbance supervision and measured/geometric overall IRI; unknown labels remain masked. See `dataset.py` and `train_multitask.patch_targets` for the exact label policy.

## Splits and coverage

The existing Kaggle protocol uses purged chronological held-out Larisa intervals; LiRA holds out roads M3/M13 for validation/test; RoadSens is TRAIN-only. Windows and extra streaming context stay inside their recording and split. `stride=1` changes overlapping training windows, not split membership. Validation is used for model/configuration selection; historical repeated use of TEST is a limitation of these research results.

Kaggle has disturbance annotations but no measured IRI. LiRA supplies section IRI but no disturbance labels or gyros. Synthetic targets come from the generator, not acceleration-derived pseudo-labels. Available supervision determines which head receives a loss.

## Obtaining artifacts

The simplest handoff is an existing prepared corpus. Preserve its manifest, record folders, and checksums. No dataset or simulator implementation is included in this source-only repository.

The included conversion tools have explicit prerequisites:

1. `python -m road_training.tools.prepare_data --repository /path/to/source-workspace --output /path/to/new-corpus` consumes the previous workspace's converted Kaggle NPZ files, purged `data/prepared/manifest.json`, and pinned synthetic exports. It is not a raw Kaggle downloader.
2. `python -m road_training.tools.prepare_lira --data-root /path/to/corpus --source-root /path/to/lira-sources` adds LiRA targets and sensor records.
3. `prepare_overall_roughness` adds preserved synthetic section IRI where the original geometry targets are available.
4. `prepare_roadsens --base-root ... --source-root ... --archive ... --output ...` adds RoadSens IMU records; `align_roadsens --base-root ... --output ...` creates the aligned corpus used by the current ensemble.

Use `--help` and the importer source to inspect exact pinned filenames before running a conversion. Some preparation steps add files to the supplied corpus; use a fresh output copy when preserving an earlier experiment.

Training/inference from prepared arrays does not require original raw files. Full replay export also needs the native sensor/GPS files named in each record's `source_file`. On a different machine, update those paths in a new manifest copy while preserving source content hashes; any old experiment manifest receipt is then intentionally invalid. Already-exported replay data does not require the original corpus or weights.
