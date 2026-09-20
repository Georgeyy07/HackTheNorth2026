"""Balance families and 100-m blocks while retaining exact source/task slots."""
from collections import defaultdict
from pathlib import Path
import numpy as np
from torch.utils.data import Sampler


def stratum(record):
    if record['source'] == 'synthetic':
        return 'synthetic/' + record['observation_domain']
    return 'real/' + record.get('dataset', 'kaggle')


def block_ids(dataset, record_index, centers):
    """Reference sections for LiRA, native station for simulation, GPS odometry otherwise."""
    record = dataset.records[record_index]
    folder = dataset.root / record['path']
    arrays = dataset._open(record_index)
    if record['source'] == 'synthetic':
        station = np.load(folder / 'target_station.npy', mmap_mode='r')
        positions = supported_centers(centers, np.isfinite(station), dataset.window_size)
        return np.floor(station[positions] / 100).astype(np.int64)
    if record.get('dataset') == 'lira_cd':
        known = ((arrays['roughness_section'] >= 0) & np.isfinite(arrays['overall_iri'])
                 & arrays['mask'].any(-1))
        positions = supported_centers(centers, known, dataset.window_size)
        # A boundary window belongs to the nearest *supported section inside
        # that window*. This is a sampling group, never an imputed centre label.
        return np.asarray(arrays['roughness_section'][positions], dtype=np.int64)
    speed = np.asarray(arrays['x'][:, 6])
    valid = np.asarray(arrays['mask'][:, 6])
    distance = np.cumsum(np.where(valid, np.maximum(speed, 0), 0), dtype=np.float64) / 100
    return (distance[centers] // 100).astype(np.int64)


def supported_centers(centers, valid, window_size):
    """Find nearest valid support within each window; never create extra groups."""
    known = np.flatnonzero(valid)
    if not len(known):
        raise ValueError('Eligible window has no supported spatial grouping')
    right = np.searchsorted(known, centers)
    a, b = known[np.maximum(right-1, 0)], known[np.minimum(right, len(known)-1)]
    positions = np.where(abs(a-centers) <= abs(b-centers), a, b)
    start = centers-window_size//2
    if np.any((positions < start) | (positions >= start+window_size)):
        raise ValueError('Spatial grouping must use support inside the input window')
    return positions


class BlockSampler:
    """Uniform family -> block -> recording/view -> eligible start.

    Selection changes sampling probabilities only. Every returned index is an
    original eligible TRAIN window; patch labels still obey their own masks.
    """
    def __init__(self, dataset, indices):
        if dataset.split != 'train':
            raise ValueError('Block sampling is TRAIN-only')
        indices = np.asarray(indices, dtype=np.int64)
        self.ends = np.asarray(dataset._ends)
        self.strata = [stratum(r) for r in dataset.records]
        tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        record_index = np.searchsorted(self.ends, indices, side='right')
        summaries = []
        for i, record in enumerate(dataset.records):
            eligible = indices[record_index == i]
            if not len(eligible):
                continue
            previous = self.ends[i-1] if i else 0
            centers = (eligible - previous) * dataset.stride + dataset.window_size // 2
            blocks = block_ids(dataset, i, centers)
            family = '|'.join(record['groups'])
            for block in np.unique(blocks):
                key = (int(block), record['id'] if block < 0 else '')
                tree[self.strata[i]][family][key].append(eligible[blocks == block])
            summaries.append(dict(recording=record['id'], family=family, stratum=self.strata[i],
                                  blocks=len(np.unique(blocks)), eligible_windows=len(eligible)))
        self.tree = {s: [list(blocks.values()) for blocks in families.values()] for s, families in tree.items()}
        self.summary = summaries

    def sample(self, template, seed):
        """Keep every baseline batch slot's real/synthetic/device/task stratum."""
        template = np.asarray(template, dtype=np.int64)
        record_index = np.searchsorted(self.ends, template, side='right')
        slots = np.asarray(self.strata)[record_index]
        output = np.empty_like(template)
        for number, name in enumerate(sorted(self.tree)):
            rng = np.random.default_rng(np.random.SeedSequence([seed, number, 411]))
            locations = np.flatnonzero(slots == name)
            families = self.tree[name]
            chosen_family = rng.integers(len(families), size=len(locations))
            for family_id, blocks in enumerate(families):
                use = locations[chosen_family == family_id]
                chosen_block = rng.integers(len(blocks), size=len(use))
                for block_id, views in enumerate(blocks):
                    current = use[chosen_block == block_id]
                    chosen_view = rng.integers(len(views), size=len(current))
                    for view_id, eligible in enumerate(views):
                        target = current[chosen_view == view_id]
                        output[target] = rng.choice(eligible, size=len(target), replace=True)
        assert np.array_equal(slots, np.asarray(self.strata)[np.searchsorted(self.ends, output, side='right')])
        return output


class CurrentIndices(Sampler):
    """Reuse persistent data workers while changing the next training block."""
    def __init__(self):
        self.indices = []

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)
