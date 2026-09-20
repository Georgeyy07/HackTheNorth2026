"""Paired real/synthetic sampling shared by research experiments."""
import numpy as np

def sample_indices(real_pool, synthetic_pool, *, seed, epoch, steps, batch_size, synthetic_per_batch):
    """Separate RNG streams keep the real sequence identical across mixed arms."""
    if not 0 <= synthetic_per_batch <= batch_size or min(steps, batch_size) < 1:
        raise ValueError("Invalid sampling budget")
    nr = batch_size - synthetic_per_batch
    if (nr and not len(real_pool)) or (synthetic_per_batch and not len(synthetic_pool)):
        raise ValueError("Missing requested training source")
    real_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 100]))
    synthetic_rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, 200]))
    real = real_rng.choice(real_pool, (steps, nr), replace=True) if nr else np.empty((steps, 0), int)
    synthetic = synthetic_rng.choice(synthetic_pool, (steps, synthetic_per_batch), replace=True) if synthetic_per_batch else np.empty((steps, 0), int)
    return np.concatenate([real, synthetic], axis=1).reshape(-1)

def pools(dataset, selected):
    indices = np.asarray(selected.indices, np.int64)
    record_indices = np.searchsorted(dataset._ends, indices, side="right")
    real = np.array([r["source"] == "real" for r in dataset.records])[record_indices]
    return indices[real], indices[~real]
