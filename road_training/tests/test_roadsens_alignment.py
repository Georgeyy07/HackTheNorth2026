"""Frame conversion must preserve geometry, labels, splits and source files."""
import copy

import numpy as np
import pytest

from road_training.dataset import CHANNELS, KAGGLE_CLASSES, LABELS, RoadDataset
from road_training.tools.align_roadsens import ROTATION, align, rotate_imu
from road_training.tools.prepare_data import checksum, read_json, write_json, write_record


def make_source(root):
    root.mkdir()
    rows = []
    for index, (dataset, split) in enumerate([
            ('roadsens4m', 'train'), ('kaggle', 'train'), ('kaggle', 'val'),
            ('kaggle', 'test'), ('lira_cd', 'train'), ('synthetic', 'train')]):
        rng = np.random.default_rng(index)
        x = rng.normal(0, .1, (64, 7)).astype(np.float32)
        x[:, 1 if dataset == 'roadsens4m' else 2] += 9.81
        mask = np.ones_like(x, dtype=bool)
        if dataset == 'roadsens4m':
            mask[:, 6] = False; mask[3, :6] = False; x[~mask] = 0
        info = dict(id=f'{dataset}_{split}', dataset=dataset, split=split,
                    source='synthetic' if dataset == 'synthetic' else 'real')
        labels = np.full((len(x), 5), -100, dtype=np.int16)
        labels[10:20, 1] = 1
        rows.append(write_record(root, info, x, mask, np.arange(len(x))/100,
            labels, np.full(len(x), np.nan, np.float32),
            dict(input_frame='recorded device', input_columns=list(CHANNELS[:6]))))
    manifest = dict(version=2, sample_rate_hz=100, channels=CHANNELS, label_columns=LABELS,
        label_classes=dict(kaggle_type=KAGGLE_CLASSES), records=rows,
        sources_policy='Test fixture.', train_statistics={})
    write_json(root/'manifest.json', manifest)
    return manifest


def test_known_axis_mapping_is_proper_rotation_for_both_sensors():
    x = np.array([[1., 2., 3., 4., 5., 6., 17.]], np.float32)
    result, mask = rotate_imu(x, np.ones_like(x, dtype=bool))
    np.testing.assert_array_equal(result, [[-3., -1., 2., -6., -4., 5., 17.]])
    assert mask.all() and np.linalg.det(ROTATION) == 1
    np.testing.assert_array_equal(ROTATION @ ROTATION.T, np.eye(3))
    for offset in (0, 3):
        np.testing.assert_array_equal(result[:, offset:offset+3], x[:, offset:offset+3] @ ROTATION.T)
        np.testing.assert_array_equal(result[:, offset:offset+3] @ ROTATION, x[:, offset:offset+3])


def test_mask_permuted_with_its_axis_without_inventing_missing_components():
    x = np.arange(14, dtype=np.float32).reshape(2, 7)
    mask = np.ones_like(x, dtype=bool)
    mask[0, 0] = False; mask[1, 5] = False; mask[:, 6] = False
    x[~mask] = np.nan
    result, observed = rotate_imu(x, mask)
    assert not observed[0, 1] and not observed[1, 3] and not observed[:, 6].any()
    assert np.isfinite(result).all() and not result[~observed].any()
    assert observed.sum() == mask.sum()


def test_dataset_alignment_preserves_sources_targets_splits_and_loader(tmp_path):
    source, output = tmp_path/'source', tmp_path/'aligned'
    original = make_source(source)
    snapshots = {str(p.relative_to(source)): checksum(p) for p in source.rglob('*') if p.is_file()}
    old = copy.deepcopy(original)
    audit = align(source, output)
    aligned = read_json(output/'manifest.json')
    assert len(audit['recordings']) == 1
    assert audit['unchanged_held_out_recordings'] == 2
    for before, after in zip(old['records'], aligned['records']):
        assert (before['id'], before['split'], before['samples']) == (after['id'], after['split'], after['samples'])
        a, b = source/before['path'], output/after['path']
        if before['dataset'] != 'roadsens4m':
            assert a.resolve() == b.resolve()
            assert before['file_sha256'] == after['file_sha256']
        else:
            for name in ('labels.npy', 'iri.npy', 'time.npy'):
                assert checksum(a/name) == checksum(b/name)
            x = np.load(b/'x.npy'); mask = np.load(b/'mask.npy')
            assert np.median(x[mask[:, 2], 2]) > 9.
            assert not mask[:, 6].any()
    assert snapshots == {str(p.relative_to(source)): checksum(p) for p in source.rglob('*') if p.is_file()}
    train_ids = {r['id'] for r in old['records'] if r['split'] == 'train'}
    for entry in [*aligned['train_statistics'].values(),
                  *[v for d in aligned['train_statistics_by_real_dataset'].values() for v in d.values()]]:
        assert set(entry['recording_ids']) <= train_ids
    dataset = RoadDataset(output, source='real', split='train', real_dataset='roadsens',
                          window_size=32, stride=1, return_labels=True)
    assert len(dataset) == 33
    assert dataset[0]['x'].shape == (32, 7)
    with pytest.raises(FileExistsError):
        align(source, output)
    with pytest.raises(ValueError, match='already aligned'):
        align(output, tmp_path/'twice')


def test_unexpected_mount_fails_without_publishing_partial_dataset(tmp_path):
    source, output = tmp_path/'source', tmp_path/'aligned'
    manifest = make_source(source)
    record = manifest['records'][0]
    path = source/record['path']/'x.npy'
    x = np.load(path); x[:, [1, 2]] = x[:, [2, 1]]
    np.save(path, x, allow_pickle=False)
    record['file_sha256']['x.npy'] = checksum(path)
    write_json(source/'manifest.json', manifest)
    with pytest.raises(ValueError, match='Unexpected RoadSens mount'):
        align(source, output)
    assert not output.exists()
