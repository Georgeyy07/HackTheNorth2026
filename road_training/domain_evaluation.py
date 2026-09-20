"""Fixed-checkpoint patch evaluation; no threshold or checkpoint selection."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from road_training.dataset import RoadDataset
from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.train_multitask import supervised_windows, run_epoch
from road_training.common import sha, write


def evaluate(checkpoint,output,source='synthetic',split='val',observation_domain=None, *, encoder_class=PatchTST,model_class=PatchTSTRoadModel):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False);config=saved['config'];root=Path(config['data_root'])
    if sha(root/'manifest.json')!=config['manifest_sha256']:raise ValueError('Dataset changed since training')
    data=RoadDataset(root,source=source,split=split,stride=1024,return_labels=True)
    selected,_,_=supervised_windows(data,16)
    if observation_domain is not None:
        if source!='synthetic' or observation_domain not in ('kaggle','lira'):
            raise ValueError('Observer subgroups require a synthetic Kaggle/LiRA view')
        indices=np.asarray(selected.indices);records=np.searchsorted(data._ends,indices,side='right')
        keep=np.array([r.get('observation_domain')==observation_domain for r in data.records])[records]
        selected=Subset(data,indices[keep].tolist())
        if not len(selected):raise ValueError('No eligible observer-domain windows')
    model=model_class(encoder_class(**config['encoder_config']));model.load_state_dict(saved['model_state'])
    model.to('cuda').eval();torch.set_num_threads(4)
    loader=DataLoader(selected,batch_size=256,num_workers=2,pin_memory=True)
    args=config['loss'];result=run_epoch(model,loader,'cuda',precision='bf16',gamma=args['gamma'],
        alpha=torch.tensor(args['alpha'],device='cuda'),roughness_weight=args['roughness_weight'],disturbance_weight=args['disturbance_weight'])
    write(output,dict(source=source,split=split,observation_domain=observation_domain,checkpoint_sha256=sha(checkpoint),epoch=saved['epoch'],windows=len(selected),metrics=result))
    return result
