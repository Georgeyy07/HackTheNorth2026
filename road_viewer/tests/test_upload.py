import time

from fastapi.testclient import TestClient

from road_training.infer_csv import export_drive
from road_training.tests.test_infer_csv import FakeOrdinal, recording
from road_viewer.server import create_app


def fake_inference(*args, **kwargs):
    return export_drive(*args, model=FakeOrdinal(), **kwargs)


def test_upload_to_inference_replay_and_persistence(tmp_path):
    csv=tmp_path/'my_drive.csv'; recording(csv)
    uploads=tmp_path/'uploads'
    app=create_app(tmp_path/'missing_export',uploads=uploads,infer_drive=fake_inference)
    with TestClient(app) as client:
        catalog=client.get('/api/catalog').json()
        assert catalog['uploads_enabled'] and not catalog['sessions']
        result=client.post('/api/import?filename=my_drive.csv',content=csv.read_bytes())
        assert result.status_code==200
        job_id=result.json()['job_id']
        for _ in range(300):
            status=client.get('/api/import/'+job_id).json()
            if status['status'] in ('complete','failed'): break
            time.sleep(.01)
        assert status['status']=='complete',status
        replay=client.get('/api/session/'+job_id).json()
        assert replay['session']['quality_mode']=='ordinal'
        assert replay['profile']['config']['q_over_r']==3.2
        assert len(replay['signals']['data'])==167
        assert any(r['quality_probability'] is not None for r in replay['updates'])
        assert all(r['iri_m_per_km'] is None for r in replay['updates'])
        assert client.get('/api/updates/'+job_id).status_code==200
    with TestClient(create_app(tmp_path/'missing_export',uploads=uploads)) as client:
        assert client.get('/api/catalog').json()['sessions'][0]['session_id']==job_id
        assert client.get('/api/session/'+job_id).status_code==200


def test_upload_validation_and_missing_job(tmp_path):
    with TestClient(create_app(tmp_path/'missing',uploads=tmp_path/'uploads',infer_drive=fake_inference)) as client:
        assert client.post('/api/import',content=b'').status_code==400
        assert client.post('/api/import?columns=bad',content=b'a').status_code==400
        assert client.post('/api/import?time_unit=auto',content=b'a').status_code==400
        assert client.get('/api/import/unknown').status_code==404
