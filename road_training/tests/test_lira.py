"""LiRA sensor timing, units, missing gyros, and independent roughness labels."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

AVAILABLE=all(importlib.util.find_spec(name) for name in ("h5py","pandas","scipy"))
if AVAILABLE:
    from road_training.tools.prepare_lira import hold_observations, project_reference, prepare_pass, supported_reference


@unittest.skipUnless(AVAILABLE,"Optional LiRA preparation dependencies required")
class LiraTests(unittest.TestCase):
    def test_reference_tail_outside_gps_support_is_not_extrapolated(self):
        stations=np.arange(12)*10.
        gps_stations=np.arange(101,dtype=float)
        gps=np.column_stack([gps_stations/111195.,np.zeros(101)])
        iri,reference=supported_reference(stations,np.ones((12,2)),gps_stations,gps)
        self.assertEqual(len(iri),11)
        self.assertEqual(len(reference),11)
        np.testing.assert_allclose(reference[-1],gps[-1])
        with self.assertRaisesRegex(ValueError,"missing the start"):
            supported_reference(stations,np.ones((12,2)),gps_stations[1:],gps[1:])

    def test_future_values_and_large_gaps_never_become_observed_inputs(self):
        query=np.array([-.1,0.,.05,.2,.99,1.,1.05])
        observations=np.array([[0.,3.],[1.,100.]])
        values,valid,source=hold_observations(query,observations,.1)
        np.testing.assert_array_equal(valid,[False,True,True,False,False,True,True])
        np.testing.assert_array_equal(values[:,0],[0,3,3,0,0,100,100])
        changed=observations.copy();changed[-1,1]=-999
        after,_,_=hold_observations(query,changed,.1)
        np.testing.assert_array_equal(values[query<1],after[query<1])
        self.assertTrue((source[valid]<=query[valid]).all())

    def test_reference_projection_is_geometric(self):
        gps=np.column_stack([np.arange(5)*10/111195.,np.zeros(5)])
        query=np.array([[15/111195.,1/111195.],[23/111195.,-2/111195.]])
        station,distance=project_reference(gps,query)
        np.testing.assert_allclose(station,[15,23],atol=1e-6)
        np.testing.assert_allclose(distance,[1,2],atol=1e-6)

    def test_pass_keeps_iri_out_of_inputs_and_leaves_disturbance_unknown(self):
        t=np.arange(301)/50
        acceleration=np.column_stack([t,np.zeros(len(t)),np.zeros(len(t)),np.ones(len(t))])
        speed=np.column_stack([t,np.full(len(t),72.)])
        gps=np.column_stack([np.arange(7),np.zeros(7),np.arange(7)*20/111195.])
        reference=np.column_stack([np.arange(21)*10/111195.,np.zeros(21)])
        group={"acc.xyz":acceleration,"obd.spd_veh":speed,"gps_mapmatch":gps}
        provenance=dict(sensors=dict(path="test.hdf5",sha256="fixture"),references=[])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            first,_=prepare_pass(root/"one","CPH1","train","/GM/1/pass_1",group,np.ones((21,2)),reference,provenance)
            second,_=prepare_pass(root/"two","CPH1","train","/GM/1/pass_1",group,np.full((21,2),3.),reference,provenance)
            a,b=root/"one"/first['path'],root/"two"/second['path']
            x=np.load(a/"x.npy");mask=np.load(a/"mask.npy")
            np.testing.assert_array_equal(x,np.load(b/"x.npy"))
            np.testing.assert_allclose(x[:,2],9.80665,atol=1e-6)
            np.testing.assert_allclose(x[:,6],20.)
            self.assertFalse(mask[:,3:6].any());self.assertTrue((x[:,3:6]==0).all())
            self.assertTrue((np.load(a/"labels.npy")==-100).all())
            iri1,iri2=np.load(a/"overall_iri.npy"),np.load(b/"overall_iri.npy")
            self.assertTrue(np.isfinite(iri1).any())
            np.testing.assert_array_equal(iri1[np.isfinite(iri1)],1.)
            np.testing.assert_array_equal(iri2[np.isfinite(iri2)],3.)
            clock=np.load(a/"time.npy");source=np.load(a/"sensor_source_time.npy")
            self.assertTrue((np.nan_to_num(source-clock[:,None],nan=0)<=0).all())


if __name__=="__main__":unittest.main()
