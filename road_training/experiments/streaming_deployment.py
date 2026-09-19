"""Evaluate true recording streams, rolling teachers and inference latency.

Run only after streaming_study finishes, so training cannot contaminate timing.
All model and seed choices are made from saved validation metrics.
"""
import gc
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from road_training.experiments.ensemble_teachers import OUT
from road_training.experiments.instance_study import DATA
from road_training.dataset import RoadDataset
from road_training.streaming_evaluation import load_teachers, load_student, validation_loader, evaluate, f1
from road_training.common import read, write, sha


class PredictionReplay(nn.Module):
    """Feed already-computed, correctly ordered predictions to common metrics."""
    def __init__(self, outputs, device='cuda'):
        super().__init__()
        self.anchor=nn.Parameter(torch.zeros(1,device=device),requires_grad=False)
        self.outputs=outputs;self.cursor=0

    def forward(self,x,mask):
        lo=self.cursor;self.cursor+=len(x)
        return {k:v[lo:self.cursor].to(x.device) for k,v in self.outputs.items()}


@torch.no_grad()
def streaming_predictions(model,data):
    predictions={}
    for i,record in enumerate(data.records):
        arrays=data._open(i);n=len(arrays['x'])//16
        pieces=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            state=model.initial_state()
            for j in range(0,n,64):
                x=torch.from_numpy(np.array(arrays['x'][j*16:min(j+64,n)*16],copy=True))[None].cuda()
                mask=torch.from_numpy(np.array(arrays['mask'][j*16:min(j+64,n)*16],copy=True))[None].cuda()
                output,state=model.stream(x,mask,state)
                pieces.append({k:v[0].float().cpu() if v.dtype!=torch.bool else v[0].cpu() for k,v in output.items()})
        emitted={k:torch.cat([p[k] for p in pieces]) for k in pieces[0]}
        delay=model.delay_patches
        aligned={k:torch.zeros_like(v) for k,v in emitted.items()}
        for k,v in emitted.items():
            if delay:aligned[k][:-delay]=v[delay:]
            else:aligned[k]=v
        predictions[record['id']]=aligned
    return predictions


@torch.no_grad()
def rolling_teacher_predictions(model,data):
    """Exactly one 64-patch window ending two patches after each target.

    Teachers attend bidirectionally only inside this already-observed window.
    Every target is emitted once, at window position 61. Inference batching
    below is solely evaluation acceleration; latency uses batch size one.
    """
    predictions={}
    for i,record in enumerate(data.records):
        arrays=data._open(i);n=len(arrays['x'])//16
        x=torch.from_numpy(np.array(arrays['x'][:n*16],copy=True))
        mask=torch.from_numpy(np.array(arrays['mask'][:n*16],copy=True))
        x=torch.cat((x.new_zeros(61*16,7),x))
        mask=torch.cat((torch.zeros(61*16,7,dtype=torch.bool),mask))
        windows=x.unfold(0,1024,16).permute(0,2,1)
        masks=mask.unfold(0,1024,16).permute(0,2,1)
        output={k:torch.zeros(n,dtype=torch.bool if k=='patch_valid' else torch.float32)
                for k in ('roughness','disturbance_logit','patch_valid')}
        for start in range(0,len(windows),128):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                result=model(windows[start:start+128].cuda(),masks[start:start+128].cuda())
            end=start+len(result['roughness'])
            for k,v in result.items():output[k][start:end]=v[:,61].to(output[k])
        predictions[record['id']]=output
    return predictions


def evaluate_cached(predictions,split,delay_seconds):
    data,loader=validation_loader(split)
    windows={k:[] for k in ('roughness','disturbance_logit','patch_valid')}
    keys=[]
    for batch in loader:
        for record,start in zip(batch['recording_id'],batch['start'].tolist()):
            keys.append((record,start));lo=start//16
            for k in windows:windows[k].append(predictions[record][k][lo:lo+64])
    model=PredictionReplay({k:torch.stack(v) for k,v in windows.items()})
    result=evaluate(model,split,loader=loader,data=data)
    assert model.cursor==len(keys)
    for row in result['events']:
        row['matched_alert_minus_reference_onset_seconds']=[
            row['predicted_intervals'][i][0]+.16+delay_seconds-row['reference_intervals'][j][0]
            for i,j in row['pairs']]
    result['emission_delay_after_patch_end_seconds']=delay_seconds
    result['inference']='Continuous recording state' if delay_seconds!=.32 else 'See associated method receipt'
    return result


def timing(function,device,repetitions=200,warmup=30):
    for _ in range(warmup):function()
    if device=='cuda':torch.cuda.synchronize()
    values=[]
    for _ in range(repetitions):
        if device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter_ns();function()
        if device=='cuda':torch.cuda.synchronize()
        values.append((time.perf_counter_ns()-start)/1e6)
    return dict(median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
                p99_ms=float(np.percentile(values,99)),mean_ms=float(np.mean(values)),repetitions=repetitions)


def bytes_in_state(state):
    return sum(v.numel()*v.element_size() for v in [state['normalizer'],*state['blocks']])


@torch.inference_mode()
def benchmark(student_path):
    report=dict(hardware=dict(platform=platform.platform(),cpu=platform.processor(),
        cuda=torch.cuda.get_device_name(),torch=torch.__version__),
        methodology='Synchronized wall clock; B=1; 30 warmups + 200 timed calls; includes model normalization and Python overhead; GPU weights FP32/autocast BF16; CPU FP32; no disk/sensor/network I/O',
        sample_period_ms=10,patch_period_ms=160,measurements=[])
    data=RoadDataset(DATA,source='real',real_dataset='kaggle',split='val',return_labels=True)
    sample=data[0];host_x=sample['x'][None];host_mask=sample['mask'][None]
    for device,threads in [('cuda',4),('cpu',1),('cpu',4)]:
        torch.set_num_threads(threads)
        teacher=load_teachers(OUT/'teacher_checkpoints.json',device)
        student=load_student(student_path,device)
        x,mask=host_x.to(device),host_mask.to(device)
        with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
            for name,model in [('single_bidirectional',teacher.models[0]),('ensemble_four',teacher)]:
                measurements=timing(lambda:model(x,mask),device)
                report['measurements'].append(dict(model=name,device=device,threads=threads,
                    input_samples=1024,**measurements))
            state=student.initial_state()
            patch=x[:,:16];pm=mask[:,:16]
            def step():
                nonlocal state
                _,state=student.stream(patch,pm,state)
            before=bytes_in_state(state)
            measurements=timing(step,device)
            assert bytes_in_state(state)==before
            report['measurements'].append(dict(model='causal_stream_one_patch',device=device,threads=threads,
                input_samples=16,state_bytes=before,**measurements))
            for patches in (64,256,1024):
                xx=x.repeat(1,patches//64,1);mm=mask.repeat(1,patches//64,1)
                stats=timing(lambda:student(xx,mm),device,repetitions=30,warmup=5)
                report['measurements'].append(dict(model='causal_full_sequence',device=device,threads=threads,
                    input_patches=patches,input_samples=patches*16,**stats))
            if device=='cuda':
                for name,model,xx,mm in [('ensemble_four',teacher,x,mask),('causal_full_window',student,x,mask)]:
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                    before_memory=torch.cuda.memory_allocated()
                    output=model(xx,mm);torch.cuda.synchronize()
                    report.setdefault('cuda_allocations',[]).append(dict(model=name,
                        additional_peak_bytes=torch.cuda.max_memory_allocated()-before_memory))
                    del output
                # Sensor-sized input upload plus prediction result download.
                hpatch=host_x[:,:16].contiguous();hmask=host_mask[:,:16].contiguous()
                def transfer_step():
                    nonlocal state
                    output,state=student.stream(hpatch.cuda(),hmask.cuda(),state)
                    return output['roughness'].cpu(),output['disturbance_logit'].cpu()
                report['measurements'].append(dict(model='causal_stream_with_transfers',device=device,threads=threads,
                    **timing(transfer_step,device)))
        del teacher,student,x,mask,state;gc.collect();torch.cuda.empty_cache()
    report['student_checkpoint']=str(student_path)
    report['student_sha256']=sha(student_path)
    return report


def run():
    plan=read(OUT/'deployment_plan.json')
    assert sha(__file__)==plan['implementation_sha256']
    assert read(OUT/'student_progress.json')['state']=='complete'
    torch.set_num_threads(4)
    finalists=read(OUT/'evaluation_plan.json')['finalists']
    validation={arm:[read(OUT/f'students/{arm}_seed{s}/roughness/val.json') for s in (62,63,64)] for arm in finalists}
    method=max(finalists,key=lambda a:np.mean([f1(m) for m in validation[a]]))
    seed=max((62,63,64),key=lambda s:f1(read(OUT/f'students/{method}_seed{s}/roughness/val.json')))
    student_path=OUT/f'students/{method}_seed{seed}/roughness/best.pt'
    write(OUT/'deployment_selection.json',dict(method=method,seed=seed,checkpoint=str(student_path),
        criterion='Best mean validation method, then best validation seed; no test selection',sha256=sha(student_path)))
    # Timing precedes the long throughput-oriented evaluations.
    write(OUT/'latency.json',benchmark(student_path))
    teacher=load_teachers(OUT/'teacher_checkpoints.json')
    student=load_student(student_path)
    for split in ('val','test'):
        data=RoadDataset(DATA,source='real',split=split,stride=1024,return_labels=True)
        for name,model,fn,delay in [('rolling_ensemble',teacher,rolling_teacher_predictions,.32),
                                  ('continuous_student',student,streaming_predictions,student.delay_patches*.16)]:
            predictions=fn(model,data)
            result=evaluate_cached(predictions,split,delay)
            result['inference']=name
            write(OUT/f'{name}_{split}.json',result)
            np.savez_compressed(OUT/f'{name}_{split}_predictions.npz',**{
                f'{record}__{key}':value.numpy() for record,values in predictions.items() for key,value in values.items()})
            print(name,split,'F1',f1(result),'section IRI',result['sections']['mae'],flush=True)
    write(OUT/'deployment_complete.json',dict(state='complete',checkpoint=str(student_path)))


if __name__=='__main__':
    run()
