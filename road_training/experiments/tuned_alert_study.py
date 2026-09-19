"""VAL-only refinement of light/adaptive Kalman filters with recall safeguards."""
from pathlib import Path
import gzip
import json

import numpy as np
import pandas as pd

from road_training.tuned_alert_filter import process, make_processor
from road_training.experiments.alert_filter_study import OUT as PREVIOUS, EXPORT, DATA, truth, evaluate
from road_training.export_test_drives import REPO, read, sha, write_json


OUT = REPO / 'reports/alert_filter_refinement_20260919'
BASE = dict(original=True, onset=.6, offset=.5)


def candidates():
    filters = [dict(gain=a, space=space) for space in ('probability','logit') for a in (.8,.85,.9,.93,.96,.98)]
    filters += [dict(gain=a,space='probability',jump=jump,fast_gain=fast)
                for a in (.5,.65,.8,.9) for jump in (.1,.2,.3) for fast in (.9,.97)]
    return [dict(**f,onset=on,offset=round(on-gap,3)) for f in filters
            for on in (.54,.56,.58,.60,.62,.64,.66,.68,.70,.72)
            for gap in (.05,.1,.15,.2)]


def acceptable(m,b):
    if not (m['events']['recall'] >= b['events']['recall']-.05
            and m['patch_recall'] >= b['patch_recall']-.05
            and m['events']['precision'] >= b['events']['precision']
            and m['events']['f1'] >= b['events']['f1']
            and m['patch_f1'] >= b['patch_f1']):
        return False
    return all(m['recall_by_type'][kind]['recall'] >= b['recall_by_type'][kind]['recall']-.1
               for kind in ('bump','depression'))


def rank(row):
    m=row['metrics']
    return (m['events']['f1'],m['patch_f1'],m['events']['precision'],m['patch_precision'])


def main():
    OUT.mkdir(exist_ok=False)
    plan=dict(purpose='Prior VAL study found no tested smoother satisfying recall guards. Add lighter gains, lower/finer thresholds, and causal innovation-adaptive Q. No new model training or IRI changes.',
              previous_selection_sha256=sha(PREVIOUS/'selection.json'),
              validation_predictions_sha256=sha(PREVIOUS/'val_predictions.json'),
              source_sha256={str(p):sha(p) for p in [Path(__file__),Path(__file__).resolve().parents[2] / 'road_training/tuned_alert_filter.py']},
              baseline=BASE,candidates=candidates(),test_previously_exposed=True,
              selection='Maximize VAL event F1, then patch F1, event precision, patch precision. Require event and patch recall within 5 percentage points of original; event precision, event F1, patch F1 no worse; bump/depression recall within 10 points. On each temporal half additionally require event recall within 10 points and patch F1 within .03 of original.',
              adaptation='Scalar random walk, Q/R=gain^2/(1-gain). Increase Q for current absolute probability innovation >=jump. No future samples. Joseph covariance update. Provisional revisions do not advance state.')
    write_json(OUT/'plan.json',plan)
    rows=read(PREVIOUS/'val_predictions.json')
    record=next(r for r in read(DATA/'manifest.json')['records'] if r['dataset']=='kaggle' and r['split']=='val')
    gt=truth(record);baseline=evaluate(process(rows,BASE),gt)
    midpoint=record['samples']/200
    def halves(processed):
        return [evaluate([r for r in processed if r['is_final'] and (r['start_s']<midpoint)==first],gt) for first in (True,False)]
    baseline_halves=halves(process(rows,BASE))
    results=[]
    for i,config in enumerate(plan['candidates']):
        filtered=process(rows,config);metrics=evaluate(filtered,gt)
        eligible=acceptable(metrics,baseline);half_scores=None
        if eligible:
            half_scores=halves(filtered)
            eligible=all(m['events']['recall']>=b['events']['recall']-.1 and m['patch_f1']>=b['patch_f1']-.03 for m,b in zip(half_scores,baseline_halves))
        results.append(dict(config=config,metrics=metrics,acceptable=eligible,halves=half_scores))
        if (i+1)%240==0:print(f"VAL candidates {i+1}/{len(plan['candidates'])}; accepted {sum(r['acceptable'] for r in results)}",flush=True)
    write_json(OUT/'validation_search.json',dict(baseline=baseline,baseline_halves=baseline_halves,candidates=results))
    valid=[r for r in results if r['acceptable']]
    winner=max(valid,key=rank) if valid else dict(config=BASE,metrics=baseline,acceptable=True,halves=baseline_halves)
    decision=dict(selected=winner,eligible_candidates=len(valid),candidate_count=len(results),test_read_this_run=False,
                  plan_sha256=sha(OUT/'plan.json'),validation_search_sha256=sha(OUT/'validation_search.json'))
    write_json(OUT/'selection.json',decision)
    print('Frozen VAL winner: '+json.dumps(winner),flush=True)
    test_record=next(r for r in read(DATA/'manifest.json')['records'] if r['dataset']=='kaggle' and r['split']=='test')
    test_rows=pd.read_parquet(EXPORT/test_record['id']/'patches.parquet').to_dict('records');test_gt=truth(test_record)
    baseline_test=evaluate(process(test_rows,BASE),test_gt)
    winner_test=evaluate(process(test_rows,winner['config']),test_gt)
    result=dict(baseline=dict(validation=baseline,test=baseline_test),tuned=dict(config=winner['config'],validation=winner['metrics'],test=winner_test),
                default_eligible_on_test=acceptable(winner_test,baseline_test),selection_sha256=sha(OUT/'selection.json'))
    write_json(OUT/'results.json',result)
    outputs=[]
    for session in read(EXPORT/'manifest.json')['sessions']:
        state=make_processor(winner['config']);dest=OUT/'tuned_updates'/(session['session_id']+'.jsonl.gz');dest.parent.mkdir(exist_ok=True)
        count=0
        with gzip.open(EXPORT/session['session_id']/'updates.jsonl.gz','rt') as src,gzip.open(dest,'wt') as dst:
            for line in src:
                row=json.loads(line);out=state.update(row)
                assert out['available_s']==row['available_s'] and out['iri_m_per_km']==row['iri_m_per_km']
                dst.write(json.dumps(out,allow_nan=False,separators=(',',':'))+'\n');count+=1
        outputs.append(dict(session=session['session_id'],updates=count,sha256=sha(dest)))
    write_json(OUT/'complete.json',dict(completed=True,selection_sha256=sha(OUT/'selection.json'),derived_updates=outputs))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
