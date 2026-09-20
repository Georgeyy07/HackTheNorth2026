"""Parse RoughRoute JSONL into causal VQF vehicle-frame samples.

Ported from the existing validated road_training.infer_jsonl reader.
"""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import numpy as np
import pandas as pd

def vqf_vehicle_sample(record, device_forward):
    """Level recorded VQF vectors and resolve arbitrary yaw using the phone mount.

    quaternion is device-to-earth, w/x/y/z. Project the mounted car's forward
    direction onto earth's horizontal plane at EACH sample so turns do not
    rotate the model's forward/left axes. No future samples or GPS are needed.
    """
    q = np.asarray(record.get('quaternion'), dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .001:
        raise ValueError('VQF requires a finite unit device-to-earth quaternion [w,x,y,z]')
    q = q/np.linalg.norm(q)
    forward = device_forward + 2*np.cross(q[1:], np.cross(q[1:], device_forward) + q[0]*device_forward)
    length = np.linalg.norm(forward[:2])
    if length < 1e-6:
        raise ValueError('Cannot determine vehicle heading: mounted forward axis points vertically')
    fx, fy = forward[:2]/length
    earth_to_vehicle = np.array([[fx, fy, 0.], [-fy, fx, 0.], [0., 0., 1.]])

    def rotate(field, required=False):
        value = record.get(field)
        if value is None and not required:
            return np.full(3, np.nan)
        vector = np.asarray(value, dtype=float)
        if vector.shape != (3,) or np.isinf(vector).any():
            raise ValueError(f'VQF requires {field} with three finite or null values')
        # A missing component cannot safely contribute to a rotated triad.
        return earth_to_vehicle @ vector if np.isfinite(vector).all() else np.full(3, np.nan)

    return rotate('earthAccel', required=True), rotate('earthGyro')


def read_recording(path, *, acceleration_frame='vqf_vehicle'):
    """Use recorded 100-Hz samples, VQF orientation, speedMps and GPS receipt times.

    The fused stream is already sampled at 100 Hz. Snap numerical timestamp
    drift to that grid; keep absent grid samples and explicit nulls missing.
    Resolve earthAccel's arbitrary yaw from the recorded mount and quaternion.
    fixed_mount explicitly restores the older input[:3] acceleration path.
    """
    if acceleration_frame not in ('vqf_vehicle', 'fixed_mount'):
        raise ValueError('acceleration_frame must be vqf_vehicle or fixed_mount')
    records = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f'Invalid JSON on line {line_number}: {error.msg}') from error
            if not isinstance(record, dict):
                raise ValueError(f'Expected a JSON object on line {line_number}')
            records.append(record)
    headers = [r for r in records if r.get('type') == 'header']
    if len(headers) != 1 or headers[0].get('schema') != 'roughroute.vehicle-imu.v1':
        raise ValueError('Expected one roughroute.vehicle-imu.v1 header')
    header = headers[0]
    if (header.get('sampleRateHz') != 100 or header.get('axes') != ['forward','left','up']
            or header.get('accelerationIncludesGravity') is not True
            or header.get('channels') != ['accel_x','accel_y','accel_z','speed']
            or header.get('units') != ['m/s²','m/s²','m/s²','m/s']):
        raise ValueError('Expected 100 Hz, forward/left/up acceleration with gravity in m/s² and speed in m/s')
    use_vqf = acceleration_frame == 'vqf_vehicle'
    if use_vqf:
        mount = np.asarray(header.get('deviceToVehicle'), dtype=float)
        if mount.size != 9:
            raise ValueError('VQF requires the header deviceToVehicle 3x3 mounting rotation')
        mount = mount.reshape(3, 3)
        if (not np.isfinite(mount).all() or not np.allclose(mount @ mount.T, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(mount), 1., atol=1e-5)):
            raise ValueError('deviceToVehicle must be a proper orthonormal rotation')
        device_forward = mount[0]
    fused = [r for r in records if r.get('type') == 'stabilized']
    if not fused:
        raise ValueError('No stabilized samples in JSONL')
    if len({r.get('segment') for r in fused}) != 1:
        raise ValueError('Import one continuous fusion segment at a time; split recordings at sensor resets')
    times = np.array([r['time'] for r in fused], dtype=float)
    receipt = np.array([r['availableAt'] for r in fused], dtype=float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError('Stabilized sensor timestamps must be finite and strictly increasing')
    if not np.isfinite(receipt).all() or (np.diff(receipt) < 0).any():
        raise ValueError('Stabilized availability timestamps must be finite and nondecreasing')
    relative = times-times[0]
    if relative[-1] > 3600:
        raise ValueError('Import one drive of at most 60 minutes at a time')
    indices = np.rint(relative*100).astype(int)
    if np.max(np.abs(relative-indices/100)) > .0001 or (np.diff(indices) <= 0).any():
        raise ValueError('Expected a stabilized stream on a 100-Hz grid (within 0.1 ms)')
    n = int(indices[-1])+1
    if n < 16:
        raise ValueError('At least 160 ms of input is required')
    origin_ms = float(receipt[0])
    grid = np.arange(n)/100
    values = np.full((n, 4), np.nan)
    gyro = np.full((n, 3), np.nan)
    available = grid.copy()
    derived = np.full(n, None, dtype=object)
    speed_timestamp = np.full(n, np.nan)
    settling = np.zeros(n, dtype=bool)
    for i, r, received in zip(indices, fused, receipt):
        if len(r.get('input', [])) != 4 or len(r.get('mask', [])) != 4:
            raise ValueError('Each stabilized sample needs four input values and four mask flags')
        if any(type(v) is not bool for v in r['mask']):
            raise ValueError('Input mask flags must be boolean')
        if 'speedMps' not in r:
            raise ValueError('Each stabilized sample needs speedMps (null when missing)')
        # input[3] and raw GPS coords.speed are deliberately not used.
        if use_vqf:
            acceleration, gyro[i] = vqf_vehicle_sample(r, device_forward)
            if not all(r['mask'][:3]):
                acceleration[:] = np.nan
        else:
            acceleration = np.asarray(r['input'][:3], dtype=float)
        data = np.r_[acceleration, np.array(r['speedMps'], dtype=float)]
        if np.isinf(data).any() or (np.isfinite(data[3]) and data[3] < 0):
            raise ValueError('Sensor values must not be infinite; speedMps must be nonnegative or null')
        observed = np.asarray(r['mask'], dtype=bool) & np.isfinite(data)
        values[i] = np.where(observed, data, np.nan)
        if not use_vqf and r.get('gyro') is not None:
            g = np.asarray(r['gyro'], dtype=float)
            if g.shape != (3,) or np.isinf(g).any():
                raise ValueError('Recorded gyro must contain three finite or null values')
            gyro[i] = g
        available[i] = max(grid[i], (received-origin_ms)/1000)
        derived[i] = r.get('speedDerived')
        speed_timestamp[i] = r.get('speedTimestamp') or np.nan
        settling[i] = bool(r.get('settling'))
    available = np.maximum.accumulate(available)
    mask = np.isfinite(values)
    if not mask[:,:3].all(-1).any():
        raise ValueError('No complete accelerometer observations')
    samples = pd.DataFrame(dict(time_s=grid, input_available_s=available,
        accel_x=values[:,0], accel_y=values[:,1], accel_z=values[:,2], speed=values[:,3],
        gyro_x=gyro[:,0], gyro_y=gyro[:,1], gyro_z=gyro[:,2],
        speed_derived=pd.array(derived, dtype='boolean'), speed_timestamp_unix_ms=speed_timestamp,
        sensor_timestamp_s=times[0]+grid, settling=settling))
    fixes = []
    rejected_gps = 0
    for r in records:
        if r.get('type') != 'location':
            continue
        event = r['event']; coords = event['coords']
        lat, lon = coords.get('latitude'), coords.get('longitude')
        source, received = float(event['timestamp']), float(r['receivedAt'])
        if (lat is None or lon is None or not np.isfinite([lat,lon,source,received]).all()
                or abs(lat) > 90 or abs(lon) > 180 or not 0 <= received-source <= 3000):
            rejected_gps += 1
            continue
        fixes.append(dict(time_s=max(0., (received-origin_ms)/1000), latitude_deg=lat,
            longitude_deg=lon, source_timestamp_unix_ms=source, received_at_unix_ms=received,
            source_age_at_receipt_s=(received-source)/1000, accuracy_m=coords.get('accuracy')))
    gps = pd.DataFrame(fixes, columns=['time_s','latitude_deg','longitude_deg',
        'source_timestamp_unix_ms','received_at_unix_ms','source_age_at_receipt_s','accuracy_m'])
    gps = gps.sort_values('time_s', kind='stable').reset_index(drop=True)
    x = np.where(mask, values, 0.).astype(np.float32)
    metadata = dict(source_schema=header['schema'], speed_source='stabilized.speedMps (m/s)',
        acceleration_frame=acceleration_frame,
        acceleration_source=('stabilized.earthAccel rotated by current mounted heading to forward/left/up, includes gravity'
            if use_vqf else 'stabilized.input[:3], forward/left/up, includes gravity'),
        gyro_source=('stabilized.earthGyro rotated to the same forward/left/up frame (display only)'
            if use_vqf else 'stabilized.gyro (display only)'), source_stabilized_samples=len(fused),
        missing_grid_samples=n-len(fused), speed_available_samples=int(mask[:,3].sum()),
        speed_derived_samples=sum(r.get('speedDerived') is True for r in fused),
        gps_rejected_fixes=rejected_gps, record_counts=dict(Counter(r.get('type') for r in records)),
        clock_alignment='First stabilized sensor sample anchored to its availableAt Unix timestamp; '
            'GPS uses receipt time; inference also waits for recorded sensor availability',
        speed_missing_policy='Preserve explicit nulls and false masks; no extra hold or GPS derivation')
    if use_vqf:
        metadata.update(device_to_vehicle=mount.tolist(),
            orientation_filter=header.get('orientationFilter'),
            heading_alignment='Per-sample quaternion rotates mounted forward axis into earth XY; '
                'left = up cross forward; vertical = earth Z; arbitrary VQF yaw cancels',
            vqf_settling_samples=int(settling.sum()),
            vqf_settling_policy='Keep recorded samples and settling flags; startup orientation can be less reliable')
    origin_ns = int(Decimal(str(fused[0]['availableAt']))*1_000_000)
    return (samples, gps, x, mask, origin_ns), metadata
