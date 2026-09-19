"""Evaluate frozen four-member quality ensembles; choose using VAL only."""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from road_training.common import read, write, sha
from road_training.instance_model import InstancePatchTST
from road_training.ordinal import OrdinalRoadModel, iri_classes
from road_training.experiments.ordinal_study import data_loader, evaluate, ARMS


class QualityEnsemble(nn.Module):
    def __init__(self, models, mode):
        super().__init__()
        if not models or any(m.mode != mode for m in models):
            raise ValueError('An ensemble requires members with the same quality mode')
        self.models, self.mode = nn.ModuleList(models), mode

    def forward(self, x, mask):
        values = [model(x, mask) for model in self.models]
        score = torch.stack([o['quality_score'].float() for o in values]).mean(0)
        probability = torch.stack([o['quality_probability'].float() for o in values]).mean(0)
        cls = (iri_classes(score) if self.mode == 'regression' else
               (probability[..., 1:].sum(-1) >= .5).long()+(probability[..., 2] > .5).long())
        detector = torch.stack([o['disturbance_logit'].float().sigmoid() for o in values]).mean(0)
        return dict(quality_score=score, quality_probability=probability, quality_class=cls,
                    disturbance_logit=torch.logit(detector.clamp(1e-6, 1-1e-6)),
                    patch_valid=torch.stack([o['patch_valid'] for o in values]).all(0))


def models_for(name, plan, study):
    models = []
    mode = 'regression' if name in ('original_regression', 'regression', 'regression_pvs') else 'ordinal'
    for seed in plan['seeds']:
        if name.startswith('original_'):
            path = Path(plan['initial_checkpoints'][str(seed)]['path'])
            saved = torch.load(path, map_location='cpu', weights_only=True)
            model = OrdinalRoadModel.from_regression(saved, mode=mode)
        else:
            path = study/f'{name}_seed{seed}'/'best.pt'
            saved = torch.load(path, map_location='cpu', weights_only=False)
            model = OrdinalRoadModel(InstancePatchTST(**saved['config']['encoder_config']),
                                     **saved['config']['model_config'])
            model.load_state_dict(saved['model_state'])
        models.append(model.cuda().eval())
    return QualityEnsemble(models, mode).cuda().eval()


def paired_block_interval(original, candidate, repeats=2000):
    """Conditional uncertainty on this road: resample 500-m spatial blocks.

    Adjacent 100-m sections share environment; do not pretend patches are
    independent samples. This still cannot estimate unseen-road uncertainty.
    """
    rows = original['sections']['rows']
    other = candidate['sections']['rows']
    assert [(r['road'], r['section'], r['truth']) for r in rows] == [(r['road'], r['section'], r['truth']) for r in other]
    keys = sorted(set((r['road'], r['section']//5) for r in rows))
    blocks = [np.array([i for i, r in enumerate(rows) if (r['road'], r['section']//5) == key]) for key in keys]
    y = np.array([r['truth'] for r in rows])
    p = np.array([r['prediction'] for r in rows]); q = np.array([r['prediction'] for r in other])
    classes = np.unique(y)
    def f1(indices, prediction):
        yy, pp = y[indices], prediction[indices]
        return np.mean([2*np.sum((yy == c) & (pp == c))/max(1, np.sum(yy == c)+np.sum(pp == c)) for c in classes])
    rng = np.random.default_rng(901)
    differences = []
    for _ in range(repeats):
        ix = np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))])
        differences.append(f1(ix, q)-f1(ix, p))
    return dict(blocks=len(blocks), block_length_m=500, repetitions=repeats,
                delta_macro_f1=float(candidate['sections']['macro_f1_present']-original['sections']['macro_f1_present']),
                percentile_95=np.quantile(differences, [.025, .975]).tolist(),
                scope='Conditional on these checkpoints and this road; not uncertainty over unseen roads/vehicles')


def run(study):
    out = study/'ensemble_evaluation'
    out.mkdir(exist_ok=False)
    plan, frozen = read(study/'plan.json'), read(study/'frozen.json')
    for name, digest in frozen['checkpoints'].items():
        assert sha(study/name/'best.pt') == digest
    torch.set_num_threads(4)
    names = ['original_regression', 'original_ordinal_pooling', *ARMS]
    results = {name: {} for name in names}
    validation = data_loader(plan['data'], 'val')
    for name in names:
        model = models_for(name, plan, study)
        results[name]['val'] = evaluate(model, *validation)
        del model
        torch.cuda.empty_cache()
    # A concrete selection receipt exists before the ensemble TEST comparison.
    ranked = sorted(names, key=lambda n: (results[n]['val']['sections']['macro_f1_present'],
        -results[n]['val']['sections']['mean_absolute_class_error']), reverse=True)
    write(out/'selection.json', dict(ranked=ranked, selected=ranked[0],
          criterion='LiRA VAL ensemble spatial-section macro F1, class MAE tie-breaker',
          ensemble_test_read_at_selection=False, prior_project_test_exposure=True,
          frozen_checkpoints=frozen['checkpoints']))
    testing = data_loader(plan['data'], 'test')
    for name in names:
        model = models_for(name, plan, study)
        results[name]['test'] = evaluate(model, *testing)
        del model
        torch.cuda.empty_cache()
        write(out/f'{name}.json', results[name])
    intervals = {name: paired_block_interval(results['original_regression']['test'], results[name]['test'])
                 for name in names if name != 'original_regression'}
    write(out/'results.json', results)
    write(out/'test_block_bootstrap.json', intervals)
    print({name: {s: round(value[s]['sections']['macro_f1_present'], 4) for s in ('val', 'test')}
           for name, value in results.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    run(args.study)
