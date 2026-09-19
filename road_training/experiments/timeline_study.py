"""Select timeline post-processing on VAL; freeze before evaluating TEST."""
from pathlib import Path
import argparse
from collections import defaultdict
import time
import numpy as np
from road_training.timeline import filter_series
from road_training.experiments.timeline_cache import OUT
from road_training.streaming_evaluation import validation_loader
from road_training.train_multitask import patch_targets
from road_training.experiments.instance_study import DATA
from road_training.metrics import scored_events, intervals, match_events, section_metrics
from road_training.common import read, write, sha


def initialize():
    OUT.mkdir(exist_ok=True,parents=True)
    if (OUT/'plan.json').exists():raise FileExistsError('Preserve existing plan')
    write(OUT/'plan.json',dict(delay_patches=2,patch_seconds=.16,
        fusion=['latest','mean','recent'],thresholds=[[.5,.5],[.55,.55],[.6,.6],[.5,.4],[.55,.45],[.6,.5]],
        roughness_alpha=[1.,.5,.2],
        defect_selection='Highest VAL event F1 subject to patch F1 >= baseline-.01 and precision >= baseline-.02; ties prefer patch F1 then fewer false alerts. Baseline included.',
        roughness_selection='Lowest equal-section VAL IRI MAE with patch MAE <= 1.02*baseline; ties prefer less smoothing',
        semantics='No future beyond two completed patches. Immutable final patches. Hysteresis decisions use only previous state/current finalized score. IRI smoothing independent of labels/sections. Unknown sensor patches reset state.',
        limitations=['160 ms patch resolution; no exact within-patch boundary claim','IRI is section-level supervision, not instantaneous surface truth',
            'Post-processing selected on one validation road; TEST historically exposed','Context votes are correlated; spread is not a confidence interval'],
        source_sha256={str(p.resolve()):sha(p) for p in [
            Path(__file__),Path(__file__).resolve().parents[2] / 'road_training/timeline.py',
            Path(__file__).with_name('timeline_cache.py')]}))


def truth(split):
    data,loader=validation_loader(split)
    rows={}
    for i,record in enumerate(data.records):
        arrays=data._open(i);n=len(arrays['x'])//16
        row=dict(record=record,time=np.array(arrays['time'][:n*16:16]),
            d=np.zeros(n,np.int64),dv=np.zeros(n,bool),r=np.zeros(n,np.float32),rv=np.zeros(n,bool),
            section=np.full(n,-1,np.int64),meters=np.zeros(n))
        speed=np.asarray(arrays['x'][:n*16,6]).reshape(-1,16)
        mask=np.asarray(arrays['mask'][:n*16,6]).reshape(-1,16).all(1)
        row['meters']=speed.mean(1)*.16*mask
        rows[record['id']]=row
    for batch in loader:
        target={k:v.numpy() for k,v in patch_targets(batch,16).items()}
        for i,(record,start) in enumerate(zip(batch['recording_id'],batch['start'].tolist())):
            row=rows[record];sl=slice(start//16,start//16+64)
            row['d'][sl]=target['disturbance'][i];row['dv'][sl]=target['disturbance_valid'][i]
            row['r'][sl]=target['roughness'][i];row['rv'][sl]=target['roughness_valid'][i]
            row['section'][sl]=batch['labels']['roughness_section'][i,::16].numpy()
    for row in rows.values():
        row['dv'][-2:]=False;row['rv'][-2:]=False
        if row['record'].get('dataset','kaggle')=='kaggle':
            annotations=read(DATA/row['record']['path']/'metadata.json')['annotations']
            row['scored']=scored_events(row['time'],row['dv'],annotations)
    return rows


def votes(split):
    z=np.load(OUT/f'{split}_votes.npz',allow_pickle=False)
    records={k.rsplit('__',1)[0] for k in z.files}
    return {r:{k:z[r+'__'+k] for k in ('probability','roughness','valid')} for r in records}


def fuse(a,mode):
    if mode=='latest':return a[:,-1].copy()
    weights=np.ones(3) if mode=='mean' else np.arange(1,4)
    return np.sum(a*weights,axis=1)/weights.sum()


def process(cache,config):
    output={}
    for record,v in cache.items():
        valid=v['valid'].all(1)
        p=fuse(v['probability'],config['defect_blend'])
        r=fuse(v['roughness'],config['roughness_blend'])
        label,iri=filter_series(p,r,valid,onset=config['onset'],offset=config['offset'],roughness_alpha=config['roughness_alpha'])
        output[record]=dict(probability=p,roughness=iri,label=label,valid=valid)
    return output


def metrics(predictions,truths):
    tp=fp=fn=0;error=[];traversals=[];events=[]
    for record,p in predictions.items():
        row=truths[record];dv=row['dv']&p['valid'];rv=row['rv']&p['valid']
        y=row['d'].astype(bool);pred=p['label']
        tp+=int((dv&y&pred).sum());fp+=int((dv&~y&pred).sum());fn+=int((dv&y&~pred).sum())
        error.extend(abs(p['roughness'][rv]-row['r'][rv]).tolist())
        for section in np.unique(row['section'][rv]):
            keep=rv&(row['section']==section)
            traversals.append(dict(road=row['record']['groups'][0],section=int(section),
                recording=record,prediction=float(p['roughness'][keep].mean()),target=float(row['r'][keep].mean())))
        if 'scored' in row:
            scored=row['scored'];valid=scored['valid']&p['valid']
            segments=intervals(row['time'],pred,valid)
            matches=match_events(segments,scored['reference'])
            km=float(row['meters'][valid].sum()/1000)
            latencies=[segments[i][0]+.48-scored['reference'][j][0] for i,j in matches['pairs']]
            events.append(dict(recording=record,**matches,distance_km=km,
                false_alerts_per_km=matches['fp']/km if km else None,
                onset_delay_median_s=float(np.median(latencies)) if latencies else None,
                predicted_intervals=segments,reference_intervals=scored['reference']))
    e_tp=sum(e['tp'] for e in events);e_fp=sum(e['fp'] for e in events);e_fn=sum(e['fn'] for e in events)
    distance=sum(e['distance_km'] for e in events)
    sections=section_metrics(traversals)
    return dict(patch_f1=2*tp/max(2*tp+fp+fn,1),precision=tp/max(tp+fp,1),recall=tp/max(tp+fn,1),
        tp=tp,fp=fp,fn=fn,event_f1=2*e_tp/max(2*e_tp+e_fp+e_fn,1),false_alerts_per_km=e_fp/distance,
        iri_patch_mae=float(np.mean(error)),iri_section_mae=sections['mae'],
        ordinal_section_f1=sections['ordinal']['macro_f1_present_classes'],events=events,sections=sections)


BASE=dict(defect_blend='latest',roughness_blend='latest',onset=.5,offset=.5,roughness_alpha=1.)


def run_validation():
    plan=read(OUT/'plan.json')
    for path,digest in plan['source_sha256'].items():assert sha(path)==digest
    cache=votes('val');gt=truth('val')
    baseline=metrics(process(cache,BASE),gt)
    write(OUT/'baseline_val.json',baseline)
    candidates=[]
    for mode in plan['fusion']:
        for on,off in plan['thresholds']:
            config=dict(BASE,defect_blend=mode,onset=on,offset=off)
            result=metrics(process(cache,config),gt)
            candidates.append(dict(config=config,metrics={k:v for k,v in result.items() if k not in ('events','sections')}))
    eligible=[c for c in candidates if c['metrics']['patch_f1']>=baseline['patch_f1']-.01
              and c['metrics']['precision']>=baseline['precision']-.02]
    selected=max(eligible,key=lambda c:(c['metrics']['event_f1'],c['metrics']['patch_f1'],-c['metrics']['false_alerts_per_km']))
    roughness=[]
    for mode in plan['fusion']:
        for alpha in plan['roughness_alpha']:
            config=dict(selected['config'],roughness_blend=mode,roughness_alpha=alpha)
            result=metrics(process(cache,config),gt)
            roughness.append(dict(config=config,metrics={k:v for k,v in result.items() if k not in ('events','sections')}))
    eligible=[c for c in roughness if c['metrics']['iri_patch_mae']<=1.02*baseline['iri_patch_mae']]
    chosen=min(eligible,key=lambda c:(c['metrics']['iri_section_mae'],-c['config']['roughness_alpha']))
    decision=dict(config=chosen['config'],validation_metrics=chosen['metrics'],test_read=False,
        plan_sha256=sha(OUT/'plan.json'),val_cache_sha256=sha(OUT/'val_votes.npz'))
    write(OUT/'selection.json',decision);write(OUT/'validation_search.json',dict(defect=candidates,roughness=roughness))
    write(OUT/'selected_val.json',metrics(process(cache,chosen['config']),gt))
    changes=[]
    for record,v in cache.items():
        row=gt[record];valid=row['dv']&v['valid'].all(1)
        if valid.any():
            p=v['probability'][valid]
            changes.append(dict(recording=record,labeled_patches=int(valid.sum()),
                context_spread_median=float(np.median(np.ptp(p,axis=1))),context_spread_p95=float(np.percentile(np.ptp(p,axis=1),95)),
                threshold_crossing_fraction=float(np.mean((p.min(1)<.5)&(p.max(1)>=.5)))))
    write(OUT/'context_variation.json',changes)
    print(decision,flush=True)


def run_test():
    selection=read(OUT/'selection.json')
    cache=votes('test');gt=truth('test')
    write(OUT/'baseline_test.json',metrics(process(cache,BASE),gt))
    write(OUT/'selected_test.json',metrics(process(cache,selection['config']),gt))
    print('TEST complete; frozen config',selection['config'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--init',action='store_true');parser.add_argument('--test',action='store_true')
    args=parser.parse_args()
    if args.init:initialize()
    elif args.test:run_test()
    else:run_validation()
