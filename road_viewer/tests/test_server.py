"""Replay API tests use tiny fixtures; real exports are tested in browser.mjs."""
import gzip
import json
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from road_viewer.server import create_app, SIGNALS


@pytest.fixture
def replay(tmp_path):
    export = tmp_path / 'export'
    export.mkdir()
    sessions = []
    for name, count in [('kaggle_fixture', 200), ('lira_fixture', 100)]:
        folder = export / name
        folder.mkdir()
        frame = pd.DataFrame({key: np.arange(count, dtype=float) / 100 for key in SIGNALS})
        if name.startswith('lira'):
            frame[['gyro_x', 'gyro_y', 'gyro_z']] = np.nan
        frame.to_parquet(folder / 'samples.parquet', index=False)
        pd.DataFrame({'time_s': [0., 1.], 'latitude_deg': [39., 39.001],
                      'longitude_deg': [22., 22.001]}).to_parquet(folder / 'gps_fixes.parquet', index=False)
        with gzip.open(folder / 'updates.jsonl.gz', 'wt') as handle:
            handle.write(json.dumps(dict(target_patch=0, available_s=.49, probability=.75,
                                         disturbance=True, is_final=True))+'\n')
        sessions.append(dict(session_id=name, samples=count, duration_s=count / 100))
    manifest = dict(sessions=sessions, samples=300, duration_s=3.)
    (export / 'manifest.json').write_text(json.dumps(manifest))
    return TestClient(create_app(export, tmp_path / 'no_profiles')), export, manifest


def test_inspector_stream_uses_small_preview_and_uncompressed_ranges(replay):
    client, export, _ = replay
    folder = export / 'kaggle_fixture'
    original = folder / 'annotated.mp4'
    preview = folder / 'annotated.preview.mp4'
    original.write_bytes(b'original-video' * 1000)
    preview.write_bytes(b'preview-video' * 200)
    url = '/demo_view/videos/kaggle_fixture.mp4'
    response = client.get(url, headers={'Range': 'bytes=0-2047', 'Accept-Encoding': 'gzip'})
    assert response.status_code == 206
    assert response.content == preview.read_bytes()[:2048]
    assert response.headers['content-length'] == '2048'
    assert response.headers['content-range'] == f'bytes 0-2047/{preview.stat().st_size}'
    assert 'content-encoding' not in response.headers
    assert 'no-transform' in response.headers['cache-control']
    assert client.head(url).headers['content-length'] == str(preview.stat().st_size)
    for path in [url + '?download=true', '/api/session/kaggle_fixture/annotated.mp4']:
        full = client.get(path, headers={'Accept-Encoding': 'gzip'})
        assert full.content == original.read_bytes()
        assert 'content-encoding' not in full.headers
    # Re-rendered recordings must never use a stale preview.
    os.utime(preview, ns=(1, 1))
    assert client.get(url).content == original.read_bytes()
    preview.unlink()
    assert client.get(url).content == original.read_bytes()


def test_catalog_is_complete_and_routes_cannot_expose_arbitrary_files(replay):
    client, _, manifest = replay
    result = client.get('/api/catalog').json()
    assert result['sessions'] == manifest['sessions']
    assert result['total_samples'] == manifest['samples']
    for path in ('/api/session/not-a-recording', '/api/session/%2E%2E%2F.env', '/.env', '/export'):
        assert client.get(path).status_code == 404
    assert client.get('/api/session/kaggle_fixture?profile=unknown').status_code == 404


def test_api_retains_every_sample_and_only_exposes_predictions_as_timed_updates(replay):
    client, export, manifest = replay
    for session in manifest['sessions']:
        folder = export / session['session_id']
        response = client.get('/api/session/' + session['session_id'])
        assert response.status_code == 200
        value = response.json()
        samples = pd.read_parquet(folder / 'samples.parquet', columns=SIGNALS)
        assert len(value['signals']['data']) == len(samples)
        reconstructed = pd.DataFrame(value['signals']['data'], columns=value['signals']['columns'], dtype=float)
        np.testing.assert_allclose(reconstructed, samples, atol=1e-12, rtol=1e-12, equal_nan=True)
        assert 'events' not in value
        assert 'probability' not in value['signals']['columns']
        with gzip.open(folder / 'updates.jsonl.gz', 'rt') as handle:
            originals = [json.loads(line) for line in handle]
        assert len(value['updates']) == len(originals)
        for actual, original in zip(value['updates'], originals):
            assert all(original.get(key) == item for key, item in actual.items())


def test_empty_update_stream_is_valid(replay):
    _, export, _ = replay
    with gzip.open(export / 'kaggle_fixture/updates.jsonl.gz', 'wt'):
        pass
    client = TestClient(create_app(export, export / 'no_profiles'))
    result = client.get('/api/session/kaggle_fixture').json()
    assert result['updates'] == [] and result['duration_s'] == 2.


def test_missing_export_gives_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match='--export'):
        create_app(tmp_path)


def test_environment_paths_and_optional_profiles_are_resolved(replay, tmp_path, monkeypatch):
    _, export, _ = replay
    filters = tmp_path / 'custom_filters'
    updates = filters / 'smoothed_updates'
    updates.mkdir(parents=True)
    profiles = [dict(id='original', label='Original', config={}),
                dict(id='smoothed', label='Smoothed', config={})]
    (filters / 'viewer_profiles.json').write_text(json.dumps(dict(profiles=profiles)))
    with gzip.open(updates / 'kaggle_fixture.jsonl.gz', 'wt') as handle:
        handle.write(json.dumps(dict(target_patch=0, available_s=.6, probability=.4,
                                    disturbance=False, is_final=True))+'\n')
    monkeypatch.setenv('ROAD_VIEWER_EXPORT', str(export))
    monkeypatch.setenv('ROAD_VIEWER_FILTERS', str(filters))
    client = TestClient(create_app())
    assert client.get('/api/catalog').json()['profiles'] == profiles
    value = client.get('/api/session/kaggle_fixture?profile=smoothed').json()
    assert value['profile']['id'] == 'smoothed'
    assert value['updates'][0]['probability'] == .4
    assert client.get('/api/updates/kaggle_fixture?profile=smoothed').status_code == 200


def test_potholes_api_crud_operations(replay):
    client, _, _ = replay
    # GET list
    response = client.get('/api/potholes')
    assert response.status_code == 200
    potholes = response.json()
    assert isinstance(potholes, list)

    # POST create new pothole
    new_pothole = {
        "latitude": 43.4723,
        "longitude": -80.5449,
        "severity": "HIGH"
    }
    create_res = client.post('/api/potholes', json=new_pothole)
    assert create_res.status_code == 200
    created = create_res.json()
    assert created['latitude'] == 43.4723
    assert created['severity'] == "HIGH"
    p_id = created['id']

    # GET with severity filter
    filtered_res = client.get('/api/potholes?severity=HIGH')
    assert filtered_res.status_code == 200
    assert any(p['id'] == p_id for p in filtered_res.json())

    # PUT update severity
    update_res = client.put(f'/api/potholes/{p_id}', json={"severity": "CRITICAL"})
    assert update_res.status_code == 200

    # DELETE pothole
    del_res = client.delete(f'/api/potholes/{p_id}')
    assert del_res.status_code == 200


def test_ordinal_probabilities_and_camera_frames_survive_export(replay):
    _, export, _ = replay
    folder = export / 'kaggle_fixture'
    with gzip.open(folder / 'updates.jsonl.gz', 'wt') as f:
        f.write(json.dumps(dict(target_patch=0, available_s=.49, is_final=True, valid=True,
                                quality_probability=[.7,.2,.1], quality_grade=0,
                                quality_name='good', provider='baseten'))+'\n')
    vision = dict(fps=1, duration_s=2, video_offset_s=.4,
                  frames=[dict(video_time_s=0,frame_index=0,detections=[])])
    (folder/'vision.json').write_text(json.dumps(vision))
    (folder/'frames').mkdir()
    (folder/'frames/000000.jpg').write_bytes(b'jpeg-fixture')
    client = TestClient(create_app(export, export/'no_profiles'))
    data = client.get('/api/session/kaggle_fixture').json()
    assert data['updates'][0]['quality_probability'] == [.7,.2,.1]
    assert data['updates'][0]['iri_m_per_km'] is None
    assert data['vision'] == vision
    assert client.get('/api/session/kaggle_fixture/frames/0').content == b'jpeg-fixture'
    assert client.get('/api/session/kaggle_fixture/annotated.mp4').status_code == 404
    (folder/'annotated.mp4').write_bytes(b'video-fixture')
    video = client.get('/api/session/kaggle_fixture/annotated.mp4')
    assert video.content == b'video-fixture' and video.headers['content-type'] == 'video/mp4'
    assert video.headers['content-disposition'].startswith('inline')
    partial = client.get('/api/session/kaggle_fixture/annotated.mp4', headers={'Range':'bytes=2-5'})
    assert partial.status_code == 206 and partial.content == b'deo-'
    assert partial.headers['content-range'] == 'bytes 2-5/13'
    assert client.get('/api/session/kaggle_fixture/annotated.mp4', headers={'Range':'bytes=-7'}).content == b'fixture'
    assert client.get('/api/session/kaggle_fixture/annotated.mp4', headers={'Range':'bytes=99-100'}).status_code == 416
    head = client.head('/api/session/kaggle_fixture/annotated.mp4')
    assert head.status_code == 200 and head.content == b'' and head.headers['content-length'] == '13'
    assert client.get('/api/session/kaggle_fixture/annotated.mp4?download=true').headers['content-disposition'].startswith('attachment')
    for path in ['kaggle_fixture/frames/-1','kaggle_fixture/frames/1','unknown/frames/0']:
        assert client.get('/api/session/'+path).status_code == 404
