import asyncio
import base64
from io import BytesIO
import sqlite3
import uuid
import httpx
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from vision_inference.client import VisionClient
from vision_inference.contract import decode_image, filter_detections
from vision_inference.service import MANIFEST, VisionService
from vision_inference.store import VisionStore


def frame(**changes):
    image = BytesIO();Image.new('RGB',(100,80),(100,100,100)).save(image,format='JPEG')
    return dict(frame_id=str(uuid.uuid4()), device_id='test-device', frame_number=1,
                timestamp=1700000000000, latitude=43.,longitude=-80.,
                gps_timestamp_ms=1700000000000,gps_received_at_ms=1700000000000,
                image=base64.b64encode(image.getvalue()).decode(),**changes)


class FakeClient:
    def __init__(self, fail=False):
        self.calls=0;self.fail=fail
    async def predict(self,data,width,height):
        self.calls+=1
        if self.fail:raise RuntimeError('unavailable')
        return dict(detections=[dict(box=[1,2,30,40],conf=.9,cls=0,label='pothole')],
                    provider='test',model_ref='test',checkpoint_sha256=MANIFEST['checkpoint_sha256'])
    async def close(self):pass


def service(tmp_path,publish=True):
    return VisionService(FakeClient(),VisionStore('sqlite:///'+str(tmp_path/'vision.db'),publish))


def counts(svc):
    with svc.store.connection() as conn:
        return tuple(conn.execute('SELECT COUNT(*) FROM '+name).fetchone()[0]
                     for name in ('vision_frames','vision_detections','potholes'))


def test_per_class_thresholds_and_validation():
    result={'detections':[dict(box=[1,2,30,40],conf=p,cls=cls) for p,cls in ((.39,0),(.4,0),(.19,1),(.2,1))]}
    kept=filter_detections(result,100,80,{0:'pothole',1:'crack'})
    assert [d['label'] for d in kept]==['pothole','crack']
    with pytest.raises(ValueError):filter_detections({'stub':True,'detections':[]},100,80,{0:'pothole'})
    with pytest.raises(ValueError):filter_detections({'detections':[dict(box=[0,0,200,40],conf=.9,cls=0)]},100,80,{0:'pothole'})
    with pytest.raises(ValueError):decode_image('invalid base64')


def test_fallback_and_credential_isolation():
    async def run():
        requests=[]
        def transport(request):
            requests.append(request)
            if request.url.host.endswith('baseten.co'):
                return httpx.Response(503,json={'error':'unavailable'})
            return httpx.Response(200,json={'detections':[dict(box=[1,2,30,40],conf=.8,cls=0)],'stub':False})
        client=VisionClient('https://model-test.api.baseten.co/production/predict','test-key','https://fallback.example/predict',
                            classes={0:'pothole'},checkpoint_sha256='hash',transport=httpx.MockTransport(transport))
        result=await client.predict(b'image',100,80)
        assert result['provider']=='fallback'
        assert requests[0].headers['authorization']=='Api-Key test-key'
        assert 'authorization' not in requests[1].headers
        assert requests[1].headers['content-type'].startswith('multipart/form-data')
        await client.close()
    asyncio.run(run())


def test_reject_stub_or_different_checkpoint():
    async def run():
        for result in ({'stub':True,'detections':[]},{'checkpoint_sha256':'wrong','detections':[]}):
            client=VisionClient(None,None,'https://fallback.example/predict',classes={0:'pothole'},
                checkpoint_sha256='expected',transport=httpx.MockTransport(lambda req:httpx.Response(200,json=result)))
            with pytest.raises(ValueError):await client.predict(b'x',100,80)
            await client.close()
    asyncio.run(run())


def test_retry_location_and_nearby_reports(tmp_path):
    async def run():
        svc=service(tmp_path);payload=frame()
        first=await svc.process(payload)
        assert await svc.process(payload)==first and svc.client.calls==1
        assert counts(svc)==(1,1,1)
        second=dict(payload,frame_id=str(uuid.uuid4()),frame_number=2,timestamp=payload['timestamp']+1000)
        assert (await svc.process(second))['pothole_ids']==first['pothole_ids']
        assert counts(svc)==(2,2,1)
        # No fresh GPS: keep the observation, never create a map pothole at invented coordinates.
        third=dict(payload,frame_id=str(uuid.uuid4()),frame_number=3,timestamp=payload['timestamp']+5000)
        response=await svc.process(third)
        assert response['latitude'] is None and response['pothole_ids']==[]
        assert counts(svc)==(3,3,1)
        with pytest.raises(ValueError,match='reused'):
            await svc.process(dict(payload,latitude=44))
    asyncio.run(run())


def test_failures_do_not_acknowledge_or_commit(tmp_path):
    async def run():
        svc=service(tmp_path);payload=frame();svc.client.fail=True
        with pytest.raises(RuntimeError):await svc.process(payload)
        assert counts(svc)==(0,0,0)
        svc.client.fail=False
        # Force a failure after the pothole insert but before the observation insert.
        execute=svc.store.execute
        def broken(conn,sql,params=()):
            if sql.startswith('INSERT INTO vision_frames'):raise RuntimeError('database failure')
            return execute(conn,sql,params)
        svc.store.execute=broken
        with pytest.raises(RuntimeError):await svc.process(payload)
        assert counts(svc)==(0,0,0)
        svc.store.execute=execute
        assert (await svc.process(payload))['status']=='ok'
        assert counts(svc)==(1,1,1)
    asyncio.run(run())


def test_cracks_stay_separate_from_potholes(tmp_path):
    async def run():
        svc=service(tmp_path)
        original=svc.client.predict
        async def cracks(*args):
            result=await original(*args)
            result['detections'][0].update(cls=1,label='crack')
            return result
        svc.client.predict=cracks
        response=await svc.process(frame())
        assert response['detections'][0]['label']=='crack' and response['pothole_ids']==[]
        assert counts(svc)==(1,1,0)
    asyncio.run(run())


def test_http_camera_and_imu_routes_share_idempotency(tmp_path):
    from road_viewer.server import create_app
    svc=service(tmp_path)
    with TestClient(create_app(vision_service=svc)) as client:
        payload=frame()
        response=client.post('/api/vision/predict',json=payload)
        assert response.status_code==200
        assert client.get('/api/vision/frames/'+payload['frame_id']).json()==response.json()
        for route in ('/ws/camera','/ws/imu'):
            with client.websocket_connect(route) as ws:
                ws.send_json(dict(payload,type='camera_frame'))
                assert ws.receive_json()==response.json()
        assert svc.client.calls==1
        assert counts(svc)==(1,1,1)
        assert client.post('/api/vision/predict',json=dict(payload,image='not an image')).status_code==422
        assert client.get('/api/vision/status').json()['enabled']
        assert not client.get('/api/imu/status').json()['enabled']


def test_concurrent_duplicate_is_one_transaction(tmp_path):
    async def run():
        svc=service(tmp_path);payload=frame()
        a,b=await asyncio.gather(svc.process(payload),svc.process(payload))
        assert a==b and counts(svc)==(1,1,1)
    asyncio.run(run())


def test_postgres_never_silently_falls_back(monkeypatch):
    import psycopg2
    monkeypatch.setattr(psycopg2,'connect',lambda *a,**kw:(_ for _ in ()).throw(psycopg2.OperationalError('down')))
    with pytest.raises(psycopg2.OperationalError):VisionStore('postgresql://unavailable/test')


def test_alert_reader_accepts_unknown_visual_severity(monkeypatch):
    from road_viewer import tiger_db
    from alert_service import potholes
    class Cursor:
        query=''
        def execute(self,query):self.query=query
        def fetchall(self):
            return [('severity',)] if 'information_schema' in self.query else [('id',43.,-80.,None)]
        def close(self):pass
    class Connection:
        def cursor(self):return Cursor()
        def close(self):pass
    monkeypatch.setattr(tiger_db,'get_db_connection',lambda:Connection())
    monkeypatch.setattr(tiger_db,'USE_POSTGRES',True)
    monkeypatch.setattr(potholes,'_POTHOLES_CACHE',None)
    assert potholes.fetch_active_potholes(force_refresh=True)[0].severity==.5
