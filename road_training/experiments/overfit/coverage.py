"""TRAIN-only sampling coverage and valid-supervision audit before learning."""
from collections import Counter,defaultdict
import numpy as np
from road_training.dataset import RoadDataset
from road_training.train_multitask import supervised_windows
from road_training.sampling import pools, sample_indices
from road_training.common import write
from road_training.experiments.overfit.study import CORPUS, OUT
from road_training.block_sampling import BlockSampler, block_ids, stratum


def main():
    data=RoadDataset(CORPUS,source='both',split='train',stride=1,return_labels=True)
    selected,_,_=supervised_windows(data,16);real,synthetic=pools(data,selected)
    baseline=sample_indices(real,synthetic,seed=42,epoch=1,steps=64,batch_size=256,synthetic_per_batch=41)
    sampler=BlockSampler(data,selected.indices);balanced=sampler.sample(baseline,4201)
    rows=[]
    for method,indices in [('uniform',baseline),('family_blocks',balanced)]:
        record_indices=np.searchsorted(data._ends,indices,side='right')
        groups=defaultdict(Counter);positive=defaultdict(list);roughness=defaultdict(list)
        for i,record in enumerate(data.records):
            chosen=indices[record_indices==i]
            if not len(chosen):continue
            previous=data._ends[i-1] if i else 0
            starts=chosen-previous;stops=starts+1024
            blocks=block_ids(data,i,starts+512);kind=stratum(record);family='|'.join(record['groups'])
            for block in blocks:groups[kind][(family,int(block))]+=1
            arrays=data._open(i);labels=np.asarray(arrays['labels'][:,1])
            pos=np.r_[0,np.cumsum(labels==1)];known=np.r_[0,np.cumsum(labels>=0)]
            positive[kind].extend(zip((pos[stops]-pos[starts]).tolist(),(known[stops]-known[starts]).tolist()))
            valid=np.isfinite(arrays['overall_iri']) if 'overall_iri' in arrays else np.zeros(len(labels),bool)
            prefix=np.r_[0,np.cumsum(valid)]
            roughness[kind].extend(((prefix[stops]-prefix[starts])/1024).tolist())
        for kind,counts in groups.items():
            values=np.array(list(counts.values()));prob=values/values.sum();pk=np.asarray(positive[kind])
            rows.append(dict(method=method,stratum=kind,sampled_windows=int(values.sum()),observed_family_blocks=len(values),
                effective_blocks=float(1/np.sum(prob**2)),known_sample_positive_fraction=float(pk[:,0].sum()/max(1,pk[:,1].sum())),
                mean_window_roughness_valid_fraction=float(np.mean(roughness[kind]))))
    def get(method):return next(r for r in rows if r['method']==method and r['stratum']=='real/lira_cd')
    assert get('family_blocks')['mean_window_roughness_valid_fraction'] >= .9 * get('uniform')['mean_window_roughness_valid_fraction']
    write(OUT/'sampling_coverage.json',dict(passed=True,train_only=True,windows_per_sampler=len(baseline),rows=rows,
        limits='Finite draw coverage, not independent road count. Label proportions use samples, not majority patch labels. Source/task/device slot sequence is identical.'))
    for row in rows:print(row)


if __name__=='__main__':main()
