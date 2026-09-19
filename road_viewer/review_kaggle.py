"""Compare the actual replay with Kaggle annotations; never tune predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from road_training.metrics import intervals, scored_events, match_events


ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'reports/test_drive_inference_20260919/kaggle_larisa_07_10_2023_test'


def binary_metrics(y, pred):
    tp = int((y & pred).sum())
    fp = int((~y & pred).sum())
    fn = int((y & ~pred).sum())
    return dict(tp=tp, fp=fp, fn=fn, precision=tp / (tp + fp), recall=tp / (tp + fn),
                f1=2 * tp / (2 * tp + fp + fn))


def main():
    samples = pd.read_parquet(FOLDER / 'samples.parquet')
    patches = pd.read_parquet(FOLDER / 'patches.parquet')
    gt = pd.read_parquet(FOLDER / 'ground_truth.parquet')
    annotations = json.loads((FOLDER / 'reference_annotations.json').read_text())['annotations']
    y = gt.defect_assumed_normal.to_numpy(dtype=int)
    final = samples.is_final.to_numpy() & samples.prediction_valid.to_numpy()
    predicted = samples.disturbance.fillna(False).to_numpy(dtype=bool)
    n = len(y) // 16
    grouped = y[:n * 16].reshape(-1, 16)
    ones, zeros = (grouped == 1).sum(1), (grouped == 0).sum(1)
    use = (ones != zeros) & patches.is_final.to_numpy()[:n] & patches.valid.to_numpy()[:n]
    patch_metrics = binary_metrics((ones > zeros)[use], patches.disturbance.fillna(False).to_numpy(bool)[:n][use])
    valid = patches.is_final.to_numpy() & patches.valid.to_numpy()
    scored = scored_events(patches.source_start_s.to_numpy(), valid, annotations)
    segments = intervals(patches.source_start_s.to_numpy(), patches.disturbance.fillna(False).to_numpy(bool), scored['valid'])
    matched = match_events(segments, scored['reference'])
    grades = (samples.loc[final, 'quality_grade'].value_counts(normalize=True).sort_index() * 100).to_dict()
    result = dict(session='kaggle_larisa_07_10_2023_test',
                  model='Mounting ensemble seeds 52–55 with frozen Timeline post-processing',
                  estimated_quality_percent={str(k): v for k, v in grades.items()},
                  quality_names=['good', 'medium', 'bad', 'terrible'],
                  median_predicted_iri=float(samples.loc[final, 'iri_m_per_km'].median()),
                  iri_ground_truth_available=False,
                  annotated_disturbance_time_percent=float(y[final].mean() * 100),
                  predicted_disturbance_time_percent=float(predicted[final].mean() * 100),
                  original_type_annotations=len(annotations), merged_binary_reference_events=len(scored['reference']),
                  predicted_events=len(segments),
                  event_metrics={k: v for k, v in matched.items() if k != 'pairs'},
                  patch_metrics=patch_metrics,
                  sample_metrics=binary_metrics(y[final].astype(bool), predicted[final]),
                  labeled_patch_ties_excluded=int((ones == zeros).sum()),
                  limitations=['This scores the complete exported timeline, not the earlier selected-window evaluation.',
                               'Event matching is one-to-one with 250 ms tolerance; unmatched alerts can include fragmentation and missing annotations.',
                               'Unannotated Kaggle samples are treated as normal under the existing weak-negative label policy.',
                               'No measured IRI exists in Kaggle; its roughness colors cannot be validated against these labels.'])
    out = Path(__file__).resolve().parent / 'test-results/kaggle-review.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
