"""Collect three past-to-future-context estimates per absolute target patch."""
import argparse
from pathlib import Path
import numpy as np
import torch
from road_training.dataset import RoadDataset
from road_training.experiments.instance_study import DATA
from road_training.streaming_evaluation import load_teachers
from road_training.experiments.ensemble_teachers import OUT as TEACHER_OUT
from road_training.common import sha, write

OUT=Path(__file__).resolve().parents[2]/'reports/timeline_20260918'


@torch.inference_mode()
def collect(split):
    OUT.mkdir(exist_ok=True,parents=True)
    destination=OUT/f'{split}_votes.npz'
    if destination.exists():return
    torch.set_num_threads(4)
    model=load_teachers(TEACHER_OUT/'teacher_checkpoints.json')
    data=RoadDataset(DATA,source='real',split=split,return_labels=True,stride=1024)
    outputs={}
    for i,record in enumerate(data.records):
        arrays=data._open(i);n=len(arrays['x'])//16
        x=torch.from_numpy(np.array(arrays['x'][:n*16],copy=True))
        mask=torch.from_numpy(np.array(arrays['mask'][:n*16],copy=True))
        # All emission windows begin on the same drive-wide patch grid.
        x=torch.cat((x.new_zeros(63*16,7),x))
        mask=torch.cat((torch.zeros(63*16,7,dtype=torch.bool),mask))
        windows=x.unfold(0,1024,16).permute(0,2,1)
        masks=mask.unfold(0,1024,16).permute(0,2,1)
        probs=np.full((n,3),np.nan,np.float32);iri=np.full_like(probs,np.nan);valid=np.zeros((n,3),bool)
        for start in range(0,n,128):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                prediction=model(windows[start:start+128].cuda(),masks[start:start+128].cuda())
            p=prediction['disturbance_logit'].float().sigmoid().cpu().numpy()
            r=prediction['roughness'].cpu().numpy();v=prediction['patch_valid'].cpu().numpy()
            ends=np.arange(start,start+len(p))
            for age in range(3):
                keep=ends>=age;targets=ends[keep]-age
                probs[targets,age]=p[keep,63-age];iri[targets,age]=r[keep,63-age];valid[targets,age]=v[keep,63-age]
        outputs.update({record['id']+'__'+k:a for k,a in dict(probability=probs,roughness=iri,valid=valid).items()})
        print(split,record['id'],n,'patches',flush=True)
    np.savez_compressed(destination,**outputs)
    write(OUT/f'{split}_cache_receipt.json',dict(sha256=sha(destination),
        teacher_receipt_sha256=sha(TEACHER_OUT/'teacher_checkpoints.json'),manifest_sha256=sha(DATA/'manifest.json'),
        split=split,delay_patches=2,columns='Future-context ages 0,1,2; row is absolute target patch; tail with no future remains NaN',
        implementation_sha256=sha(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['val','test'],default='val')
    collect(parser.parse_args().split)
