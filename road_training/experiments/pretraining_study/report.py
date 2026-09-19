"""Summarize paired downstream metrics, reconstruction histories and compute."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from road_training.common import read, write
from road_training.experiments.pretraining_study.study import OUT, check_plan


def flat_metrics(folder, split):
    patch = read(folder/f'patch_{split}.json')['metrics']
    event = read(folder/f'event_section_{split}.json')
    binary = patch['disturbance']['classification']['per_class']['disturbance']
    rows = event['disturbance']
    tp, fp, fn = [sum(r[k] for r in rows) for k in ('tp','fp','fn')]
    distance = sum(r['scored_distance_km'] for r in rows)
    return dict(precision=binary['precision'], recall=binary['recall'], f1=binary['f1'],
        patch_iri_mae=patch['roughness']['mae'], patch_iri_rmse=patch['roughness']['rmse'],
        section_iri_mae=event['roughness']['mae'], section_iri_rmse=event['roughness']['rmse'],
        event_f1=2*tp/max(2*tp+fp+fn,1), false_alerts_per_km=fp/distance if distance else None,
        average_precision=float(np.mean([r['patch_ranking']['average_precision'] for r in rows])),
        ordinal_macro_f1=event['roughness']['ordinal']['macro_f1_present_classes'])


def main():
    plan=check_plan()
    for split in ['val','test']:
        if not read(OUT/f'{split}_evaluation_complete.json')['passed']:
            raise ValueError('Complete fixed held-out evaluation first')
    summary={}; pairs={}
    for arm in plan['arms']:
        rows=[{split:flat_metrics(OUT/f'{arm}_seed{seed}',split) for split in ['val','test']} for seed in plan['seeds']]
        summary[arm]=dict(per_seed=rows)
        for split in ['val','test']:
            summary[arm][split]={key:dict(mean=float(np.mean([r[split][key] for r in rows])),
                seed_sd=float(np.std([r[split][key] for r in rows],ddof=1))) for key in rows[0][split]}
    for fine, scratch in [('droppatch_finetuned','patchtst_scratch'),('arctan_finetuned','dit_scratch')]:
        pairs[fine]={split:{key:[a[split][key]-b[split][key] for a,b in zip(summary[fine]['per_seed'],summary[scratch]['per_seed'])]
            for key in summary[fine][split]} for split in ['val','test']}
    for method in ['droppatch','arctan']:
        for variant in ['linear','low_lr']:
            treatment, control = f'{method}_{variant}', f'{method}_finetuned'
            pairs[treatment+'_minus_full']={split:{key:[a[split][key]-b[split][key] for a,b in
                zip(summary[treatment]['per_seed'],summary[control]['per_seed'])]
                for key in summary[treatment][split]} for split in ['val','test']}
    write(OUT/'comparison.json', dict(arms=summary, paired_differences=pairs, seeds=plan['seeds']))
    lines=['# Patch-reconstruction pretraining comparison','',
        'Three paired seeds. F1 is positive-class binary localized-disturbance F1; IRI errors are m/km.',
        'All weights were selected on real VAL combined loss and frozen before the final TEST comparison.',
        'Seed variability is not an estimate of cross-road generalization.','']
    for split in ['val','test']:
        lines += [f'## {split.upper()}','',
            '| Method | Precision | Recall | Patch F1 | Patch IRI MAE | 100-m IRI MAE | Event F1 | False alerts/km |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in plan['arms']:
            metrics=summary[arm][split]
            values=[f"{metrics[key]['mean']:.3f} ± {metrics[key]['seed_sd']:.3f}" for key in
                ['precision','recall','f1','patch_iri_mae','section_iri_mae','event_f1','false_alerts_per_km']]
            lines.append('| '+arm+' | '+' | '.join(values)+' |')
        lines.append('')
    lines += ['## Protocol and limitations','',
        'Pretraining uses real plus corrected synthetic TRAIN recordings without labels. DropPatch uses fixed epoch 12 '
        '(12 × 512 updates); ArcTan selects minimum validation loss over 50 × 512 updates. The user requested the '
        'DropPatch cap after seed 42 had already completed 50 epochs; its saved epoch-12 best weights are reused. '
        'The other DropPatch seeds stop at 12, keeping the original 50-epoch cosine schedule so their first 12 epochs match. '
        'Fine-tuning uses real TRAIN labels only, with identical input draws, fresh head initialization, normalization and '
        'stopping rules within each architecture pair. Both use CUDA BF16, batch 256 and LR 1e-4. '
        'Fine-tuning validation occurs every 256 updates, with early stopping.','',
        'Additional user-requested adaptation tests: linear probes freeze encoder weights and dropout, '
        'training only two linear readouts (softplus is retained as the nonnegative IRI link). '
        'Low-LR variants keep the same MLP heads and use encoder LR 5e-6 versus head LR 1e-4. '
        'Linear probes change both head capacity and encoder trainability, so they do not isolate freezing alone.','',
        'The scratch control matches its own pretrained architecture. ArcTan uses a mixed-channel DiT; '
        'DropPatch uses channel-independent PatchTST. Cross-architecture differences cannot be attributed solely to the objective.','',
        'Pretraining adds optimizer updates and unlabeled exposure. Total compute is not matched to the scratch controls. '
        'The combined TRAIN normalizer is shared by all arms, including scratch. This isolates learned weights rather than all source-data exposure.','',
        'The prior project has already evaluated these TEST roads historically. No threshold or checkpoint was selected from TEST in this comparison. '
        'VAL/TEST roughness sections are overwhelmingly good under the project bins; severe-road and pothole-specific reliability remain unproven. '
        'Reconstruction losses across methods have different definitions and scales.','',
        'Details: [frozen plan](plan.json), [checkpoint freeze](frozen_checkpoints.json), '
        '[paired results](comparison.json), [training jobs](training_complete.json).','']
    (OUT/'report.md').write_text('\n'.join(lines))
    fig,axes=plt.subplots(2,2,figsize=(17,9),layout='constrained')
    names={'patchtst_scratch':'PatchTST\nscratch','dit_scratch':'DiT\nscratch',
           'droppatch_finetuned':'DropPatch\nfull FT','arctan_finetuned':'ArcTan\nfull FT',
           'droppatch_linear':'DropPatch\nlinear','arctan_linear':'ArcTan\nlinear',
           'droppatch_low_lr':'DropPatch\n20× lower LR','arctan_low_lr':'ArcTan\n20× lower LR'}
    labels=[names[arm] for arm in plan['arms']]
    colors=['#7894a8','#2166ac','#70b4d6','#12466e','#af9c7c','#d77b22','#efb66b','#9f4708']
    n=len(labels)
    for row,split in enumerate(['val','test']):
        for col,key in enumerate(['f1','section_iri_mae']):
            ax=axes[row,col]
            means=[summary[arm][split][key]['mean'] for arm in plan['arms']]
            errors=[summary[arm][split][key]['seed_sd'] for arm in plan['arms']]
            ax.bar(range(n),means,yerr=errors,capsize=4,color=colors)
            ax.set_xticks(range(n),labels,fontsize=8); ax.set_title(f'{split.upper()}: '+('binary disturbance F1 ↑' if key=='f1' else '100-m IRI MAE ↓'))
            ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
            if key=='f1':ax.set_ylim(0,1)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,method in zip(axes,['droppatch','arctan']):
        for seed in plan['seeds']:
            history=read(OUT/f'{method}_pretrain_seed{seed}'/'history.json')
            ax.plot([r['epoch'] for r in history],[r['val']['loss'] for r in history],label=f'seed {seed}')
        ax.set(title=f'{method}: VAL reconstruction objective',xlabel='Epoch',ylabel='Method-specific loss');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'pretraining_curves.png',dpi=160);plt.close(fig)


if __name__ == '__main__': main()
