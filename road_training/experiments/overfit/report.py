"""Summarize completed ablations without reading TEST or choosing on it."""
from pathlib import Path
import numpy as np

from road_training.experiments.overfit.study import OUT, ARMS
from road_training.common import read, write, sha


def cls(metrics):
    return metrics['disturbance']['classification']['per_class']['disturbance']


def summary(values):
    return dict(mean=float(np.mean(values)), seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None,
                values=[float(x) for x in values])


def main():
    plan=read(OUT/'plan.json');rows=[]
    for arm in plan['arms']:
        for seed in plan['seeds']:
            folder=OUT/f'{arm}_seed{seed}'
            if not (folder/'progress.json').exists():continue
            progress=read(folder/'progress.json')
            if progress['state']!='complete':continue
            history=read(folder/'history.json');best=min(history,key=lambda h:h['val']['loss'])
            assert best['updates']==progress['best_updates']
            val=best['val'];train=read(folder/'clean_train_best.json');last=history[-1]
            train_last=read(folder/'clean_train_last.json')
            real_train=train['by_source']['real'];real_last=train_last['by_source']['real']
            row=dict(arm=arm,seed=seed,parameters=read(folder/'config.json')['parameters'],
                best_updates=best['updates'],stopped_updates=progress['updates'],
                val_loss=val['loss'],val_f1=cls(val)['f1'],val_precision=cls(val)['precision'],val_recall=cls(val)['recall'],
                val_patch_iri_mae=val['roughness']['mae'],val_patch_iri_rmse=val['roughness']['rmse'],
                clean_real_train_f1=cls(real_train)['f1'],clean_real_train_iri_mae=real_train['roughness']['mae'],
                clean_real_train_last_f1=cls(real_last)['f1'],clean_real_train_last_iri_mae=real_last['roughness']['mae'],
                last_val_f1=cls(last['val'])['f1'],last_val_iri_mae=last['val']['roughness']['mae'],
                last_val_loss=last['val']['loss'],elapsed_seconds=progress['elapsed_seconds'],
                checkpoint_sha256=sha(folder/'best.pt'))
            detail_path=folder/'event_section_val.json'
            if detail_path.exists():
                detail=read(detail_path)
                assert detail['checkpoint_sha256']==row['checkpoint_sha256']
                row['val_section_iri_mae']=detail['roughness']['mae']
                row['val_section_iri_rmse']=detail['roughness']['rmse']
                row['val_ordinal_macro_f1']=detail['roughness']['ordinal']['macro_f1_present_classes']
                events=detail['disturbance'];tp,fp,fn=[sum(e[k] for e in events) for k in ('tp','fp','fn')]
                row['val_event_f1']=2*tp/max(2*tp+fp+fn,1)
                row['val_false_alerts_per_km']=fp/sum(e['scored_distance_km'] for e in events)
                row['val_average_precision']=float(np.mean([e['patch_ranking']['average_precision'] for e in events]))
            rows.append(row)
    groups=[]
    metrics=['val_loss','val_f1','val_precision','val_recall','val_patch_iri_mae','clean_real_train_f1',
             'clean_real_train_iri_mae','clean_real_train_last_f1','clean_real_train_last_iri_mae',
             'last_val_f1','last_val_iri_mae','last_val_loss','best_updates','stopped_updates']
    extra=['val_section_iri_mae','val_section_iri_rmse','val_event_f1','val_false_alerts_per_km','val_average_precision','val_ordinal_macro_f1']
    for arm in plan['arms']:
        selected=[r for r in rows if r['arm']==arm]
        if not selected:continue
        keys=metrics+[k for k in extra if all(k in r for r in selected)]
        groups.append(dict(arm=arm,seeds=len(selected),parameters=selected[0]['parameters'],
                           metrics={k:summary([r[k] for r in selected]) for k in keys}))
    paired=[]
    for arm in plan['arms']:
        if arm.endswith('_base'):continue
        baseline='R_base' if arm.startswith('R_') else 'M_base'
        a={r['seed']:r for r in rows if r['arm']==arm};b={r['seed']:r for r in rows if r['arm']==baseline}
        seeds=sorted(a.keys()&b.keys())
        if seeds:
            keys=metrics+[k for k in extra if all(k in a[s] and k in b[s] for s in seeds)]
            paired.append(dict(arm=arm,baseline=baseline,seeds=seeds,
                differences={k:summary([a[s][k]-b[s][k] for s in seeds]) for k in keys}))
    write(OUT/'comparison.json',dict(rows=rows,summary=groups,paired_differences=paired,
        plan_sha256=sha(OUT/'plan.json'),complete=len(rows)==len(plan['arms'])*len(plan['seeds']),
        test_read=False,limits='Validation-only development comparison; seed SD is not uncertainty on new roads. Training probes are fixed clean evaluation-mode inputs; training/validation label distributions differ.'))
    lines=['# Overfitting ablations: validation only','',
        'Means ± sample seed SD; binary disturbance F1. Roughness is patch IRI MAE in m/km. Every arm uses the same frequent validation and early stopping. No TEST reads.','',
        '| Arm | Seeds | Parameters | VAL F1 | VAL precision | VAL IRI MAE | VAL loss | Best update |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in groups:
        cells=[]
        for key in ['val_f1','val_precision','val_patch_iri_mae','val_loss']:
            v=row['metrics'][key];cells.append(f"{v['mean']:.3f}"+(f" ± {v['seed_sd']:.3f}" if v['seed_sd'] is not None else ''))
        lines.append(f"| {row['arm']} | {row['seeds']} | {row['parameters']:,} | "+' | '.join(cells)+f" | {row['metrics']['best_updates']['mean']:.0f} |")
    lines+=['','`M_`: real + revised synthetic, 215/41 windows per batch. `R_`: real only. `base`: unchanged encoder/regularization and uniform windows. `blocks`: family/block/view balancing. `small`: 64-wide, two layers. `regularized`: dropout 0.25 and weight decay 0.05. `rotation`: shared ≤5° IMU-frame perturbation. `noise`: small held-sample jitter and window-constant bias. `combined`: all interventions.','',
        'Checkpoint selection is minimum combined real-VAL loss, not maximum F1. An improvement in one head may trade off against the other. The detailed JSON includes paired differences and clean TRAIN probes.']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    if not rows:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,len(plan['arms']),figsize=(27,7),sharex=True)
    for col,arm in enumerate(plan['arms']):
        for seed in plan['seeds']:
            path=OUT/f'{arm}_seed{seed}'/'history.json'
            if not path.exists():continue
            h=read(path);steps=[r['updates'] for r in h]
            for split,color in [('train','#3b9168'),('val','#287cba')]:
                # Match real-only TRAIN metrics to real VAL for mixed arms.
                values=[r[split]['by_source']['real'] if split=='train' else r[split] for r in h]
                axes[0,col].plot(steps,[cls(v)['f1'] for v in values],color=color,alpha=.6)
                axes[1,col].plot(steps,[v['roughness']['mae'] for v in values],color=color,alpha=.6)
        axes[0,col].set_title(arm)
        for row in range(2):axes[row,col].grid(alpha=.2)
        axes[1,col].set_xlabel('Updates')
    axes[0,0].set_ylabel('Binary F1');axes[1,0].set_ylabel('Patch IRI MAE (m/km)')
    fig.suptitle('Three seeds per arm; green: online real TRAIN metrics, blue: clean real VAL\nTRAIN uses dropout/augmentation, so use saved clean probes for evaluation-mode gaps.')
    fig.tight_layout();fig.savefig(OUT/'learning_curves.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5))
    for ax,key,title in zip(axes,['val_f1','val_precision','val_patch_iri_mae'],
                            ['Binary disturbance F1 ↑','Binary disturbance precision ↑','Patch roughness MAE (m/km) ↓']):
        x=np.arange(len(groups));values=[r['metrics'][key]['mean'] for r in groups]
        spread=[r['metrics'][key]['seed_sd'] or 0 for r in groups]
        ax.errorbar(x,values,yerr=spread,fmt='o',capsize=4,color='#287cba')
        ax.set_xticks(x,[r['arm'].replace('M_','').replace('R_','real_') for r in groups],rotation=50,ha='right')
        ax.set_title(title);ax.grid(alpha=.2)
    fig.suptitle('Real validation only: fixed checkpoint selection and threshold\nError bars are optimization seed SD, not uncertainty across new roads')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
