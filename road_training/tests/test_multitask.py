"""Two-head gradients, independent label masks and source-separated metrics."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from road_training.patchtst import PatchTST, PatchTSTRoadModel
from road_training.train_multitask import joint_loss, patch_targets, run_epoch
from road_training.tools.prepare_overall_roughness import align_sections


def example():
    return dict(x=torch.randn(2,16,7),mask=torch.ones(2,16,7,dtype=torch.bool),source=["real","synthetic"],
        labels=dict(localized_disturbance=torch.tensor([[0]*4+[1]*4+[0]*4+[-100]*4]*2),
            overall_iri=torch.tensor([[float("nan")]*16,[1.]*4+[3.]*4+[5.]*4+[7.]*4]),
            overall_iri_valid=torch.tensor([[False]*16,[True]*16]),
            roughness_section=torch.tensor([[-1]*16,[0]*4+[1]*4+[2]*4+[3]*4])))


def model():
    return PatchTSTRoadModel(PatchTST(patch_length=4,d_model=16,n_heads=2,n_layers=1,
                                    ffn_dim=32,dropout=0,max_patches=4),dropout=0)


class MultitaskTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)

    def test_one_encoder_call_two_patch_outputs_and_both_heads_backpropagate(self):
        m=model();batch=example();calls=[]
        hook=m.encoder.register_forward_hook(lambda *args:calls.append(1))
        out=m(batch['x'],batch['mask']);hook.remove()
        self.assertEqual(len(calls),1)
        for value in out.values():self.assertEqual(value.shape,(2,4))
        self.assertTrue((out['roughness']>=0).all())
        targets=patch_targets(batch,4)
        joint_loss(out,targets)['total'].backward()
        for module in [m.encoder.value_embedding,m.roughness_head[-1],m.disturbance_head[-1]]:
            self.assertGreater(module.weight.grad.abs().sum().item(),0)
        copied=model();copied.load_state_dict(m.state_dict());m.eval();copied.eval()
        for key,value in m(batch['x'],batch['mask']).items():
            torch.testing.assert_close(value,copied(batch['x'],batch['mask'])[key])

    def test_missing_roughness_does_not_train_or_decay_roughness_head(self):
        m=model();batch=example();batch['labels']['overall_iri_valid'].fill_(False)
        before=[p.detach().clone() for p in m.roughness_head.parameters()]
        optimizer=torch.optim.AdamW(m.parameters(),lr=.01,weight_decay=.1)
        optimizer.zero_grad(set_to_none=True)
        losses=joint_loss(m(batch['x'],batch['mask']),patch_targets(batch,4))
        self.assertEqual(losses['roughness'].item(),0)
        losses['total'].backward();optimizer.step()
        for value,param in zip(before,m.roughness_head.parameters()):
            self.assertIsNone(param.grad);torch.testing.assert_close(value,param)

    def test_roughness_only_batch_has_no_disturbance_head_gradient(self):
        m=model();batch=example();batch['labels']['localized_disturbance'].fill_(-100)
        loss=joint_loss(m(batch['x'],batch['mask']),patch_targets(batch,4))
        loss['total'].backward()
        self.assertTrue(all(p.grad is None for p in m.disturbance_head.parameters()))
        self.assertGreater(m.roughness_head[-1].weight.grad.abs().sum().item(),0)

    def test_section_crossings_missing_labels_and_sensor_gaps_are_masked(self):
        batch=example()
        batch['labels']['roughness_section'][1,3]=1  # First patch spans sections.
        batch['labels']['overall_iri_valid'][1,4]=False
        batch['mask'][1,8,:]=False
        targets=patch_targets(batch,4)
        self.assertEqual(targets['roughness_valid'].tolist(),[[False]*4,[False,False,False,True]])
        self.assertTrue(torch.isfinite(targets['roughness']).all())
        self.assertEqual(targets['roughness'][1,3].item(),7)

    def test_binary_focal_gamma_zero_matches_bce(self):
        batch=example();targets=patch_targets(batch,4);out=model()(batch['x'],batch['mask'])
        loss=joint_loss(out,targets,gamma=0)
        valid=targets['disturbance_valid']
        expected=torch.nn.functional.binary_cross_entropy_with_logits(out['disturbance_logit'][valid],targets['disturbance'][valid].float())
        torch.testing.assert_close(loss['disturbance'],expected)

    def test_real_roughness_is_unavailable_and_source_scores_stay_separate(self):
        m=model();batch=example();before={k:v.clone() for k,v in m.state_dict().items()}
        metrics=run_epoch(m,[batch],'cpu',precision='fp32')
        self.assertIsNone(metrics['by_source']['real']['roughness']['mae'])
        self.assertEqual(metrics['by_source']['synthetic']['roughness']['labeled_patches'],4)
        self.assertEqual(metrics['disturbance']['labeled_patches'],6)
        for key,value in before.items():torch.testing.assert_close(value,m.state_dict()[key])

    def test_kaggle_lira_and_synthetic_scores_use_only_each_available_target(self):
        original=example();index=torch.tensor([0,1,1])
        batch=dict(x=original['x'][index],mask=original['mask'][index],
                   labels={k:v[index].clone() for k,v in original['labels'].items()},
                   source=['real','real','synthetic'],dataset=['kaggle','lira','synthetic'])
        batch['labels']['localized_disturbance'][1].fill_(-100)
        batch['mask'][1,:,3:6]=False;batch['x'][1,:,3:6]=0
        metrics=run_epoch(model(),[batch],'cpu',precision='fp32')
        self.assertEqual(metrics['roughness']['labeled_patches'],8)
        self.assertEqual(metrics['disturbance']['labeled_patches'],6)
        collections=metrics['by_dataset']
        self.assertIsNone(collections['kaggle']['roughness']['mae'])
        self.assertIsNone(collections['lira']['disturbance']['classification'])
        self.assertEqual(collections['lira']['roughness']['labeled_patches'],4)
        self.assertEqual(collections['synthetic']['roughness']['labeled_patches'],4)
        self.assertTrue(np.isfinite(collections['lira']['loss']))

    def test_total_iri_alignment_includes_defects_and_rejects_ambiguous_sections(self):
        sections=[dict(background_iri_m_per_km=1.,realized_iri_m_per_km=3.),
                  dict(background_iri_m_per_km=2.,realized_iri_m_per_km=4.)]
        iri,ids=align_sections(np.array([1.,1.,2.,99.]),np.array([True,True,True,False]),sections)
        np.testing.assert_array_equal(iri[:3],[3.,3.,4.]);self.assertTrue(np.isnan(iri[3]))
        np.testing.assert_array_equal(ids,[0,0,1,-1])
        with self.assertRaisesRegex(ValueError,'uniquely'):
            align_sections(np.ones(4),np.ones(4,bool),sections+sections)
        with self.assertRaisesRegex(ValueError,'no matching'):
            align_sections(np.full(4,99.),np.ones(4,bool),sections)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),"CUDA BF16 required")
    def test_cuda_bf16_joint_training_keeps_weights_fp32(self):
        m=model().cuda();batch=example();optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4)
        before=[m.roughness_head[-1].weight.detach().clone(),m.disturbance_head[-1].weight.detach().clone()]
        metrics=run_epoch(m,[batch],'cuda',optimizer,precision='bf16')
        self.assertTrue(np.isfinite(metrics['loss']))
        self.assertTrue(all(p.dtype==torch.float32 for p in m.parameters()))
        for old,head in zip(before,[m.roughness_head,m.disturbance_head]):self.assertFalse(torch.equal(old,head[-1].weight))


if __name__=='__main__':unittest.main()
