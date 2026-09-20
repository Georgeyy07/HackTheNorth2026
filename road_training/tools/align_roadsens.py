"""Create a new dataset with RoadSens in nominal forward/left/up coordinates.

The publication describes a rear camera facing the road, and the recorded
gravity is on +device-Y. For that upright Android mount, vehicle XYZ equals
(-device-Z, -device-X, device-Y). This is a proper rotation, not a reflection.
It removes the large nominal mounting mismatch, without claiming to recover
small undocumented yaw/tilt offsets or LiRA's exact horizontal convention.
"""
import argparse
import copy
from pathlib import Path
import tempfile

import numpy as np

from road_training.dataset import collection_name
from road_training.tools.prepare_data import checksum, read_json, training_statistics, write_json

ROTATION = np.array([[0., 0., -1.], [-1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
PERMUTATION = [2, 0, 1, 5, 3, 4, 6]
SIGNS = np.array([-1., -1., 1., -1., -1., 1., 1.], dtype=np.float32)
PAPER = 'https://www.nature.com/articles/s41597-026-07072-y'


def rotate_imu(x, mask):
    """Same fixed rotation for accel/gyro; speed and missingness are preserved."""
    if x.ndim != 2 or x.shape[1] != 7 or mask.shape != x.shape or mask.dtype != bool:
        raise ValueError('Expected [samples,7] values and boolean mask')
    aligned_mask = mask[:, PERMUTATION].copy()
    aligned = x[:, PERMUTATION] * SIGNS
    aligned[~aligned_mask] = 0
    if not np.isfinite(aligned).all():
        raise ValueError('Observed RoadSens values must be finite')
    return aligned, aligned_mask


def align(base, output):
    """Transform only RoadSens; link all other arrays without modifying them."""
    base, output = Path(base).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Choose a new destination; existing data is never overwritten')
    manifest_path = base/'manifest.json'
    original_bytes = manifest_path.read_bytes()
    manifest = copy.deepcopy(read_json(manifest_path))
    if 'roadsens_alignment' in manifest:
        raise ValueError('RoadSens is already aligned; refusing a second rotation')
    roadsens = [r for r in manifest['records'] if collection_name(r) == 'roadsens']
    if not roadsens or any(r['split'] != 'train' for r in roadsens):
        raise ValueError('Expected the existing TRAIN-only RoadSens integration')
    recipe = dict(version=1, base_root=str(base), base_manifest_sha256=checksum(manifest_path),
        recipe_sha256=checksum(__file__), rotation_device_to_model=ROTATION.tolist(),
        target_axes=['nominal vehicle forward', 'nominal vehicle left', 'up'],
        mapping='(x,y,z) = (-device_z,-device_x,device_y), for both accelerometer and gyro',
        evidence=dict(publication=PAPER, mounting='Rear camera faces the road; positive device Y dominates observed gravity'),
        uncertainty='Nominal mounting conversion, not per-drive yaw/tilt calibration. LiRA horizontal convention remains unverified.',
        fitting='No fitted parameters, labels, future observations, or held-out data used to choose rotation',
        retained='Kaggle, LiRA, synthetic, timestamps, labels, splits and speed unchanged')
    audit = dict(recipe=recipe, recordings=[], unchanged_recordings=0,
                 unchanged_held_out_recordings=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.roadsens_align_', dir=output.parent) as temporary:
        staging = Path(temporary)/'dataset'
        (staging/'records').mkdir(parents=True)
        for record in manifest['records']:
            source = base/record['path']
            destination = staging/'records'/record['id']
            record['path'] = destination.relative_to(staging).as_posix()
            if collection_name(record) != 'roadsens':
                destination.symlink_to(source.resolve(), target_is_directory=True)
                audit['unchanged_recordings'] += 1
                audit['unchanged_held_out_recordings'] += record['split'] in ('val', 'test')
                continue
            for name in ('x.npy', 'mask.npy', 'metadata.json'):
                if checksum(source/name) != record['file_sha256'][name]:
                    raise ValueError(f'RoadSens source changed: {record["id"]}/{name}')
            x, mask = np.load(source/'x.npy'), np.load(source/'mask.npy')
            aligned, aligned_mask = rotate_imu(x, mask)
            complete = mask[:, :3].all(-1)
            before_median = np.median(x[complete, :3], axis=0)
            after_median = np.median(aligned[complete, :3], axis=0)
            if before_median.argmax() != 1 or before_median[1] < 5:
                raise ValueError(f'Unexpected RoadSens mount in {record["id"]}; inspect before aligning')
            destination.mkdir()
            for path in source.iterdir():
                if path.name not in ('x.npy', 'mask.npy', 'metadata.json'):
                    (destination/path.name).symlink_to(path.resolve())
            np.save(destination/'x.npy', aligned, allow_pickle=False)
            np.save(destination/'mask.npy', aligned_mask, allow_pickle=False)
            metadata = read_json(source/'metadata.json')
            metadata['original_input_frame'] = metadata['input_frame']
            metadata['original_input_columns'] = metadata['input_columns']
            metadata['input_columns'] = manifest['channels'][:6]
            metadata['input_frame'] = 'Nominal forward/left/up; fixed upright-phone conversion; total acceleration retains gravity'
            metadata['frame_alignment'] = recipe
            write_json(destination/'metadata.json', metadata)
            record['valid_fraction'] = aligned_mask.mean(0).tolist()
            record['file_sha256'] = {p.name: checksum(p) for p in sorted(destination.iterdir())}
            audit['recordings'].append(dict(id=record['id'], samples=len(x),
                acceleration_median_before=before_median.tolist(), acceleration_median_after=after_median.tolist(),
                max_accel_norm_error=float(np.abs(np.linalg.norm(x[complete, :3], axis=1)
                    -np.linalg.norm(aligned[complete, :3], axis=1)).max()),
                original_x_sha256=checksum(source/'x.npy'), aligned_x_sha256=checksum(destination/'x.npy')))
        # Keep legacy statistics truthful for all loader filters. The new trainer
        # still uses instance normalization; these global moments are not used.
        manifest['train_statistics'] = {s: training_statistics(staging, manifest['records'], s,
            allow_missing_channels=True) for s in ('real', 'synthetic', 'both')}
        manifest['train_statistics_by_real_dataset'] = {}
        for dataset in ('kaggle', 'lira', 'roadsens'):
            chosen = [r for r in manifest['records'] if r['source'] != 'real' or collection_name(r) == dataset]
            manifest['train_statistics_by_real_dataset'][dataset] = {
                s: training_statistics(staging, chosen, s, allow_missing_channels=True)
                for s in ('real', 'synthetic', 'both')}
        manifest['roadsens_alignment'] = recipe
        manifest['sources_policy'] += ' RoadSens nominal forward/left/up alignment; original source records preserved.'
        write_json(staging/'manifest.json', manifest)
        write_json(staging/'alignment_audit.json', audit)
        (staging/'manifest.before_alignment.json').write_bytes(original_bytes)
        if manifest_path.read_bytes() != original_bytes:
            raise ValueError('Source manifest changed while aligning')
        staging.rename(output)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-root', type=Path, default=Path('road_training/data_with_roadsens'))
    parser.add_argument('--output', type=Path, default=Path('road_training/data_roadsens_aligned'))
    args = parser.parse_args()
    audit = align(args.base_root, args.output)
    print(f'Aligned {len(audit["recordings"])} RoadSens recordings in {args.output}')
