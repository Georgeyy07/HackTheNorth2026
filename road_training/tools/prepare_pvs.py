"""Import left dashboard acceleration and causal GPS speed as TRAIN-only PVS.

Released roughness labels are a centered acceleration/speed proxy clustered per
vehicle. They are weak auxiliary labels, never LiRA IRI or detection negatives.
The release interpolated some IMU samples without preserving observation masks;
this importer cannot recover that missing provenance or claim live input timing.
"""
import argparse
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from road_training.common import write, sha
from road_training.tools.prepare_lira import clean_time, hold_observations


def csv(path, **kwargs):
    return pd.read_csv(path, compression='zip' if zipfile.is_zipfile(path) else None, **kwargs)


def import_record(raw, output, number):
    folder = raw/f'PVS {number}'
    imu_file, gps_file, labels_file = [folder/name for name in
        ('dataset_mpu_left.csv', 'dataset_gps.csv', 'dataset_labels.csv')]
    imu = csv(imu_file, usecols=['timestamp', 'acc_x_dashboard', 'acc_y_dashboard', 'acc_z_dashboard'])
    label_columns = ['good_road_left', 'regular_road_left', 'bad_road_left']
    # pandas usecols selects columns but preserves the CSV's original order.
    labels = csv(labels_file, usecols=label_columns)[label_columns].to_numpy()
    t = imu.timestamp.to_numpy()
    if len(t) != len(labels) or not np.allclose(np.diff(t), .01, atol=2e-5, rtol=0):
        raise ValueError('PVS labels must align exactly with the released 100-Hz sensor rows')
    if not np.isin(labels, [0, 1]).all() or not (labels.sum(1) == 1).all():
        raise ValueError('Expected exactly one weak quality label per row')
    gps = csv(gps_file, usecols=['timestamp', 'speed_meters_per_second'])
    gps = clean_time(gps[['timestamp', 'speed_meters_per_second']].to_numpy())
    gps = gps[gps[:, 1] >= 0]
    speed, speed_valid, source_time = hold_observations(t, gps, maximum_age=2.5)
    acceleration = imu[['acc_x_dashboard', 'acc_y_dashboard', 'acc_z_dashboard']].to_numpy()
    # GPS speed derivatives correlate with +Y in all nine released sessions
    # (r=.43..69), while X is weak/negative. Proper 90-degree yaw: forward=Y,
    # left=-X, up=Z. This is approximate mounting alignment, not precision INS.
    acceleration = acceleration[:, [1, 0, 2]] * np.array([1., -1., 1.])
    x = np.column_stack((acceleration, speed)).astype(np.float32)
    mask = np.isfinite(x)
    mask[:, 3] &= speed_valid
    x[~mask] = 0
    # The original heuristic sets stopped vehicles to good. Exclude this shortcut.
    quality = labels.argmax(1).astype(np.int16)
    quality[~mask[:, :3].all(1) | ~mask[:, 3] | (x[:, 3] < 1.4)] = -100
    target = output/f'pvs_{number}'
    target.mkdir(parents=True, exist_ok=False)
    for name, value in dict(x=x, mask=mask, time=t-t[0], quality=quality,
                            speed_source_time=source_time-t[0]).items():
        np.save(target/f'{name}.npy', value, allow_pickle=False)
    return dict(id=f'pvs_{number}', path=target.name, samples=len(x), split='train',
                vehicle=(number-1)//3, scenario=(number-1)%3,
                source_sha256={p.name: sha(p) for p in (imu_file, gps_file, labels_file)},
                file_sha256={p.name: sha(p) for p in target.glob('*.npy')},
                quality_counts=np.bincount(quality[quality >= 0], minlength=3).tolist(),
                missing_speed_fraction=float((~mask[:, 3]).mean()),
                excluded_quality_samples=int((quality < 0).sum()),
                mean_acceleration=acceleration.mean(0).tolist())


def prepare(raw, output):
    raw, output = Path(raw), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    records = [import_record(raw, output, i) for i in range(1, 10)]
    manifest = dict(channels=['accel_x', 'accel_y', 'accel_z', 'speed'], sample_rate_hz=100,
        records=records, source_url='https://www.kaggle.com/datasets/jefmenegazzo/pvs-passive-vehicular-sensors-datasets',
        label_source='Released vehicle-relative acceleration/speed proxy labels; not measured IRI',
        split_policy='All nine PVS sessions TRAIN-only; no PVS metric is used as evidence of physical roughness accuracy. LiRA road splits unchanged.',
        frame_policy='Approximate vehicle frame: [source Y, -source X, source Z]. +Y forward supported by GPS acceleration correlations .43-.69 in all nine TRAIN sessions; Z approximately up. Same modest mounting augmentation as existing sources.',
        speed_policy='Last observed nonnegative native GPS speed, at most 2.5 seconds old; no future interpolation',
        quality_policy='Left-side labels only; exclude missing acceleration/speed and speed below 1.4 m/s',
        license='Author repository: CC BY-NC-ND 4.0. Data remain local; do not redistribute this converted corpus.',
        source_label_notebook='https://github.com/jefmenegazzo/MPU-9250-and-GPS-Raw-Data-Pre-Processing/blob/master/src/3%20-%20Data%20Class%20Labeling.ipynb')
    write(output/'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.raw, args.output)
    print(f"Imported {sum(r['samples'] for r in result['records']):,} samples from nine TRAIN-only sessions")
