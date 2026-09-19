"""Report all declared RCD runs and historical matched PatchTST controls."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from road_training.common import read, write, sha
from road_training.experiments.pretraining_study.report import flat_metrics
from road_training.experiments.rcd_study.study import OUT, REFERENCE, check_plan


def main():
    plan=check_plan()
    for split in ['val','test']:assert read(OUT/f'{split}_evaluation_complete.json')['passed']
    summary={}
    for arm in plan['arms']:
        rows=[{split:flat_metrics(OUT/f'{arm}_seed{seed}',split) for split in ['val','test']} for seed in plan['seeds']]
        summary[arm]=dict(per_seed=rows)
        for split in ['val','test']:
            summary[arm][split]={k:dict(mean=float(np.mean([r[split][k] for r in rows])),
                seed_sd=float(np.std([r[split][k] for r in rows],ddof=1))) for k in rows[0][split]}
    assert sha(REFERENCE/'comparison.json')==plan['reference_comparison_sha256']
    reference=read(REFERENCE/'comparison.json')['arms']['patchtst_scratch']
    paired={arm:{split:{key:[a[split][key]-b[split][key] for a,b in zip(value['per_seed'],summary['rcd_scratch']['per_seed'])]
        for key in value[split]} for split in ['val','test']} for arm,value in summary.items() if arm!='rcd_scratch'}
    write(OUT/'comparison.json',dict(arms=summary,patchtst_reference=reference,paired_differences_from_rcd_scratch=paired,seeds=plan['seeds']))
    lines=['# Time-RCD road pretraining comparison','',
        'Three paired seeds; binary positive-class localized-disturbance F1. IRI errors are m/km. '
        'All checkpoints selected by real-VAL combined loss and frozen before TEST; threshold fixed at0.5.','']
    for split in ['val','test']:
        lines += [f'## {split.upper()}','',
            '| Model | Precision | Recall | Patch F1 | AP | Patch IRI MAE | 100-m IRI MAE | Event F1 | False alerts/km |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm, values in [('PatchTST scratch (previous)',reference),*summary.items()]:
            m=values[split]
            cells=[f"{m[k]['mean']:.3f} ± {m[k]['seed_sd']:.3f}" for k in
                ['precision','recall','f1','average_precision','patch_iri_mae','section_iri_mae','event_f1','false_alerts_per_km']]
            lines.append('| '+arm+' | '+' | '.join(cells)+' |')
        lines.append('')
    lines += ['## Pretraining and verification','',
        'Time-RCD uses labeled synthetic anomaly classification plus masked reconstruction. '
        'Our compact model keeps channel-patch joint attention, same/different-variate bias, RoPE, RMSNorm, '
        'gated GELU blocks, per-timestep projection, two-class anomaly head and reconstruction MLP. '
        'Masks cover15% of whole time patches across channels; CE and masked observed-value MSE each have weight1. '
        'No real road labels enter pretraining. The shared pretraining projection and heads are replaced by fresh patch-level road heads.','',
        'The corpus and fixed combined TRAIN normalization match the earlier comparison. The RCD scratch control has the identical '
        'encoder architecture, fresh heads, real TRAIN window draws and downstream optimization. Pretraining uses '
        'AdamW LR5e-4, weight decay1e-5, up to12 epochs of256 updates, batch256, CUDA BF16; synthetic VAL selects the checkpoint. '
        'Fine-tuning uses the previous fixed recipe: full LR1e-4, frozen linear LR1e-4, or encoder LR5e-6/head LR1e-4.','',
        'This is a road adaptation of the method, not a replication of the authors 2.5B-point generic synthetic corpus or full-size model. '
        'Physics-road labels do not guarantee the same context-dependent normality training distribution. '
        'The normalizer and simulator use real TRAIN information, so the pretraining is synthetic-gradient-only rather than strict zero-shot.','',
        'The historical TEST set has been evaluated before. No current threshold/checkpoint is tuned on TEST. '
        'Seed variation does not estimate cross-road generalization. Sparse severe IRI grades and uncertain pothole-specific labels remain limitations.','',
        'Sources: [paper v5](https://arxiv.org/html/2509.21190v5), '
        '[pinned official code](https://github.com/thu-sail-lab/Time-RCD/tree/372bb980426b2f67007311c6f3165ab789c79bef). '
        'The arXiv metadata currently carries a withdrawal note; the repository describes ICML2026 acceptance. '
        'This experiment assesses the released method independently of its publication claims.','',
        'Artifacts: [plan](plan.json), [checkpoint freeze](frozen_checkpoints.json), [paired results](comparison.json).','']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    arms=[('PatchTST\nscratch',reference),('RCD\nscratch',summary['rcd_scratch']),
          ('RCD\nfull FT',summary['rcd_finetuned']),('RCD\nlinear',summary['rcd_linear']),('RCD\nLR /20',summary['rcd_low_lr'])]
    for i,split in enumerate(['val','test']):
        for j,key in enumerate(['f1','section_iri_mae']):
            ax=axes[i,j]
            ax.bar(range(len(arms)),[r[split][key]['mean'] for _,r in arms],
                yerr=[r[split][key]['seed_sd'] for _,r in arms],capsize=4,color=['#888888','#7693ac','#225e9c','#7cbaa5','#5a8f40'])
            ax.set_xticks(range(len(arms)),[name for name,_ in arms]);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
            ax.set_title(f'{split.upper()}: '+('disturbance F1 ↑' if key=='f1' else '100-m IRI MAE ↓'))
            if key=='f1':ax.set_ylim(0,1)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for seed in plan['seeds']:
        h=read(OUT/f'rcd_pretrain_seed{seed}/history.json')
        axes[0].plot([r['epoch'] for r in h],[r['val']['loss'] for r in h],label=f'seed{seed}')
        axes[1].plot([r['epoch'] for r in h],[r['clean_val']['f1'] for r in h],label=f'seed{seed}')
    for ax,title in zip(axes,['Synthetic VAL CE + masked MSE ↓','Clean synthetic VAL sample F1 ↑']):
        ax.set(title=title,xlabel='Pretraining epoch');ax.grid(alpha=.2);ax.legend()
    fig.savefig(OUT/'pretraining_curves.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
