"""Replay API tests use tiny fixtures; real exports are tested in browser.mjs."""
import gzip
import json

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
