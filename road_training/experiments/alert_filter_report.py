"""Produce the fixed-threshold ablation, viewer profiles and comparison plot."""
import gzip
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from road_training.alert_filter import AlertPostprocessor
from road_training.experiments.alert_filter_study import OUT, EXPORT, DATA, BASE, process, truth, evaluate
from road_training.export_test_drives import read, sha, write_json


def main():
    results = read(OUT / 'results.json')
    decision = read(OUT / 'selection.json')
    kalman = decision['family_champions']['kalman']['config']
    threshold = dict(kind='none', space='probability', onset=kalman['onset'], offset=kalman['offset'])
    ablation = dict(config=threshold, rationale='Use exactly the VAL-chosen Kalman onset/offset without smoothing to separate threshold and smoothing effects. This is a deterministic matched ablation, not TEST parameter selection.',
                    selection_sha256=sha(OUT/'selection.json'), test_previously_viewed=True)
    write_json(OUT/'matched_threshold_ablation_plan.json', ablation)
    records = {r['split']:r for r in read(DATA/'manifest.json')['records'] if r['dataset']=='kaggle' and r['split'] in ('val','test')}
    val = read(OUT/'val_predictions.json')
    test = pd.read_parquet(EXPORT/records['test']['id']/'patches.parquet').to_dict('records')
    matched = dict(config=threshold, validation=evaluate(process(val,threshold),truth(records['val'])),
                   test=evaluate(process(test,threshold),truth(records['test'])))
    write_json(OUT/'matched_threshold_ablation.json', matched)
    profiles = dict(default='original', original_export_unchanged=True, calibration_dataset='Kaggle validation',
                    roughness_changed=False, profiles=[
        dict(id='original', label='Original · 60% / 50%', config=BASE,
             description='Original consensus and hysteresis. Highest recall of these three test modes.'),
        dict(id='threshold', label='Higher threshold · 70% / 50%', config=threshold,
             description='Stricter onset without smoothing. Fewer alerts, with reduced recall.'),
        dict(id='kalman', label='Kalman · fewer alerts, lower recall', config=kalman,
             description='Scalar Kalman smoothing plus 70% / 50% hysteresis. Test: 8 unmatched alerts, 24 missed events. Failed validation recall guard; experimental.')])
    for session in read(EXPORT/'manifest.json')['sessions']:
        for profile in profiles['profiles'][1:]:
            state = AlertPostprocessor(**profile['config'])
            target = OUT / (profile['id']+'_updates') / (session['session_id']+'.jsonl.gz')
            target.parent.mkdir(exist_ok=True)
            with gzip.open(EXPORT/session['session_id']/'updates.jsonl.gz','rt') as src,gzip.open(target,'wt') as dst:
                for line in src:
                    row=json.loads(line);filtered=state.update(row)
                    assert filtered['available_s']==row['available_s'] and filtered['iri_m_per_km']==row['iri_m_per_km']
                    dst.write(json.dumps(filtered,allow_nan=False,separators=(',',':'))+'\n')
    write_json(OUT/'viewer_profiles.json',profiles)
    comparison=dict(original=results['baseline'],higher_threshold=matched,kalman=results['kalman'],
                    ema=results['ema'],recall_guard_choice=results['selected'])
    write_json(OUT/'comparison.json',comparison)
    # Entire recording, then a fixed first-20-second example; no cherry-picked span.
    rows={name:process(test,value['config']) for name,value in comparison.items() if name in ('original','higher_threshold','kalman')}
    fig,axes=plt.subplots(2,1,figsize=(13,7),constrained_layout=True)
    colors={'original':'#718096','higher_threshold':'#19846a','kalman':'#cf734b'}
    for name,values in rows.items():
        f=[r for r in values if r['is_final']]
        t=np.array([r['start_s'] for r in f]);p=np.array([r['probability'] for r in f]);d=np.array([r['disturbance'] for r in f])
        axes[0].step(t,d.astype(float)+(0 if name=='original' else 1.25 if name=='higher_threshold' else 2.5),where='post',lw=.7,color=colors[name],label=name)
        use=t<20
        # Higher threshold leaves the original score unchanged.
        if name!='higher_threshold':axes[1].plot(t[use],p[use],lw=1.4,color=colors[name],label=name)
    for a in truth(records['test'])['annotations']:
        lo=a['rel_t_start']-truth(records['test'])['origin'];hi=a['rel_t_end']-truth(records['test'])['origin']
        if lo<20:axes[1].axvspan(lo,min(hi,20),color='#1c9e7b',alpha=.13)
    axes[0].set(yticks=[.5,1.75,3],yticklabels=['Original','Higher threshold','Kalman'],xlabel='Road target time (s)',title='Full Kaggle TEST replay: final disturbance decisions')
    axes[1].axhline(.6,color='#718096',ls=':',label='Original onset .60')
    axes[1].axhline(.7,color='#cf734b',ls='--',label='Filtered onset .70')
    axes[1].set(xlim=(0,20),ylim=(0,1.04),xlabel='Road target time (s)',ylabel='Disturbance score',title='First 20 seconds: annotation intervals shaded green')
    axes[1].legend(loc='upper right',ncol=2,fontsize=8)
    fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    lines=['# Causal alert filtering comparison','',
        'Parameters were selected using the current ensemble on the full Kaggle validation drive. TEST was already exposed in earlier work; these are diagnostic held-out-split comparisons, not a new untouched benchmark.',
        '', '| Mode | VAL precision | VAL recall | VAL event F1 | TEST alerts | TEST unmatched | TEST missed | TEST precision | TEST recall | TEST event F1 | TEST patch F1 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,item in comparison.items():
        v=item['validation']['events'];t=item['test'];e=t['events']
        lines.append(f"| {name} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {t['predicted_events']} | {e['fp']} | {e['fn']} | {e['precision']:.3f} | {e['recall']:.3f} | {e['f1']:.3f} | {t['patch_f1']:.3f} |")
    lines += ['', 'Original is onset .60 / offset .50. The matched higher threshold is .70 / .50. The Kalman and EMA profiles use .70 / .50; both have a steady gain of .80. Kalman R=1 and Q/R=3.2 per 160-ms finalized patch. EMA alpha=.80. Both smooth the probability score, not the logit, in the selected family champions.',
        '', 'The Kalman and EMA family champions failed the predeclared recall guard on validation. They are exposed as experimental comparisons, not recommended replacements. All 156 candidates were evaluated on validation; none of the tested smoothers met the guard. The feasible validation choice was wider hysteresis (.60/.40), which mainly merges fragments; on TEST it lowered unmatched alerts only 38→36 and worsened patch F1 .733→.714. Original remains the viewer default.',
        '', 'Kalman preserves 15/15 annotated manhole and 3/3 bump detections on TEST, but depression recall falls 4/4→2/4 and crack recall 48/55→33/55. These are small class counts, not proof of generalization.',
        '', 'Filtering is forward-only, constant-state, after the existing three-context consensus. There is no new lookahead or added sample buffer. Smoothing can delay threshold crossings or erase short events. The last two patches and incomplete tail stay provisional. Their previews remain unfiltered and labeled as such; provisional revisions never advance the Kalman state. IRI and GPS are unchanged.',
        '', 'The scalar random-walk Kalman update is P-=P+Q; K=P-/(P-+R); x=x+K(z-x). Joseph-form covariance is used. With fixed Q and R, the gain converges to a constant, explaining the matched EMA results. The score and covariance are not calibrated defect probabilities or reliable uncertainty estimates. The model assumptions do not repair systematic bias.',
        '', 'Implementation references: [USAF Test Pilot School scalar Kalman equations](https://usafa-ece.github.io/tps-sy6301/block04/L04_KalmanFilterScalar_Demo.html) and [NIST EWMA definition](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm).',
        '', 'The same fixed profiles are offered for LiRA playback, but LiRA has no disturbance labels and therefore cannot validate them. Kaggle roughness calibration is also unaffected.',
        '', 'Five numerical/state tests cover hand-computed Kalman updates, bounds, EMA equivalence, missing-data reset, causality, provisional revisions, and hysteresis. Source hashes and frozen decisions are in plan.json, selection.json, and complete.json. The original export and neural checkpoints were not modified.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({name:value['test']['events'] for name,value in comparison.items()},indent=2))


if __name__=='__main__':main()
