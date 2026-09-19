"""Import a user CSV and export ordinal inference for the existing web replay."""
import argparse
import gzip
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from road_training.alert_filter import DEFAULT_ALERT_FILTER
from road_training.common import write, sha
from road_training.ordinal_checkpoints import load_ordinal_ensemble
from road_training.ordinal_stream import OrdinalRoadStream

DEFAULT_ENSEMBLE = Path(__file__).resolve().parents[1]/'models/ordinal_pvs/ensemble.json'
ALIASES = dict(time_s=['time_s','timestamp','time'], accel_x=['accel_x','ax'],
               accel_y=['accel_y','ay'], accel_z=['accel_z','az'], speed=['speed','speed_mps'],
               latitude_deg=['latitude_deg','latitude','lat'], longitude_deg=['longitude_deg','longitude','lon'],
               gyro_x=['gyro_x'], gyro_y=['gyro_y'], gyro_z=['gyro_z'])


def read_recording(path, *, columns=None, time_unit='s', acceleration_unit='m/s2', speed_unit='m/s'):
    """Causal previous-sample resampling; never use a future sensor or GPS fix."""
    if time_unit not in ('s','ms','iso') or acceleration_unit not in ('m/s2','g') or speed_unit not in ('m/s','km/h'):
        raise ValueError('Unsupported timestamp, acceleration or speed unit')
    frame = pd.read_csv(path)
    if not len(frame):
        raise ValueError('CSV contains no samples')
    columns = columns or {}
    if not isinstance(columns, dict) or set(columns)-set(ALIASES) or any(not isinstance(v,str) for v in columns.values()):
        raise ValueError('Column mapping must map supported canonical names to CSV column names')
    names = {}
    for key, aliases in ALIASES.items():
        name = columns.get(key)
        if name is not None and name not in frame:
            raise ValueError(f'Mapped column is missing: {name}')
        names[key] = name if name is not None else next((a for a in aliases if a in frame), None)
    missing = [k for k in ('time_s','accel_x','accel_y','accel_z') if names[k] is None]
    if missing:
        raise ValueError('Missing required columns: '+', '.join(missing))
    if time_unit == 'iso':
        dates = pd.to_datetime(frame[names['time_s']], utc=True, errors='raise', format='ISO8601')
        if dates.isna().any():
            raise ValueError('Timestamps must not be missing')
        origin_ns = int(dates.iloc[0].value)
        t = (dates.astype('int64').to_numpy()-origin_ns)/1e9
    else:
        values = pd.to_numeric(frame[names['time_s']], errors='raise').to_numpy(dtype=float)
        scale = 1000. if time_unit == 'ms' else 1.
        t = (values-values[0])/scale
        origin_ns = int(round(values[0]/scale*1e9)) if np.isfinite(values[0]) and values[0]/scale >= 1e9 else None
    if not np.isfinite(t).all() or (np.diff(t) <= 0).any():
        raise ValueError('Timestamps must be finite and strictly increasing; one sensor sample per row')
    if t[-1] > 3600:
        raise ValueError('Import one drive of at most 60 minutes at a time')
    grid = np.arange(int(np.floor(t[-1]*100+1e-7))+1)/100
    if len(grid) < 16:
        raise ValueError('At least 160 ms of input is required')
    values = {}
    for key in ALIASES:
        if key != 'time_s':
            a = pd.to_numeric(frame[names[key]], errors='raise').to_numpy(dtype=float) if names[key] else np.full(len(t), np.nan)
            if np.isinf(a).any():
                raise ValueError(f'{key} must not contain infinity')
            values[key] = a
    for key in ('accel_x','accel_y','accel_z'):
        values[key] *= 9.80665 if acceleration_unit == 'g' else 1.
    values['speed'] /= 3.6 if speed_unit == 'km/h' else 1.
    if (values['speed'][np.isfinite(values['speed'])] < 0).any():
        raise ValueError('Speed must be nonnegative')
    aligned = {}
    for key in ('accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z','speed'):
        a = values[key]
        # Missing IMU readings break holds; sparse speed observations may hold.
        use = np.isfinite(a) if key == 'speed' else np.ones(len(t), bool)
        source_t, source_a = t[use], a[use]
        if not len(source_t):
            aligned[key] = np.full(len(grid), np.nan)
            continue
        index = np.searchsorted(source_t, grid+1e-10, side='right')-1
        clipped = np.maximum(index, 0)
        observed = (index >= 0) & (grid-source_t[clipped] <= (3. if key == 'speed' else .05)+1e-10)
        aligned[key] = np.where(observed, source_a[clipped], np.nan)
    gps_valid = np.isfinite(values['latitude_deg']) & np.isfinite(values['longitude_deg'])
    if ((np.abs(values['latitude_deg'][gps_valid]) > 90) | (np.abs(values['longitude_deg'][gps_valid]) > 180)).any():
        raise ValueError('GPS coordinates are outside latitude/longitude bounds')
    gps = pd.DataFrame(dict(time_s=t[gps_valid], latitude_deg=values['latitude_deg'][gps_valid], longitude_deg=values['longitude_deg'][gps_valid]))
    samples = pd.DataFrame(dict(time_s=grid, input_available_s=grid, **aligned))
    x = samples[['accel_x','accel_y','accel_z','speed']].to_numpy(dtype=np.float32)
    mask = np.isfinite(x)
    if not mask[:,:3].any():
        raise ValueError('No observed accelerometer samples')
    x[~mask] = 0.
    return samples, gps, x, mask, origin_ns


def export_drive(csv_path, output, *, session_id='uploaded_drive', ensemble=DEFAULT_ENSEMBLE,
                 device='auto', model=None, progress=None, display_name=None, **options):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session_id):
        raise ValueError('Session ID may contain only letters, digits, underscores and hyphens')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    samples, gps, x, mask, origin = read_recording(csv_path, **options)
    folder = output/session_id
    folder.mkdir()
    samples.to_parquet(folder/'samples.parquet', index=False)
    gps.to_parquet(folder/'gps_fixes.parquet', index=False)
    loaded_from_receipt = model is None
    if model is None:
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        model = load_ordinal_ensemble(ensemble, device=device)
    stream = OrdinalRoadStream(model)
    times = gps.time_s.to_numpy()
    count = 0
    with gzip.open(folder/'updates.jsonl.gz', 'wt') as handle:
        for start in range(0, len(x), 1024):
            for row in stream.push(x[start:start+1024], mask[start:start+1024]):
                # Target location uses the last GPS fix known by target start,
                # not a later fix seen when the delayed prediction is emitted.
                ix = int(np.searchsorted(times, row['start_s']+1e-10, side='right')-1)
                age = row['start_s']-float(times[ix]) if ix >= 0 else None
                valid = ix >= 0 and age <= 3.
                row.update(session_id=session_id, target_gps_valid=valid,
                    target_gps_fix_index=ix if valid else None, target_gps_age_s=age,
                    target_latitude_deg=float(gps.iloc[ix].latitude_deg) if valid else None,
                    target_longitude_deg=float(gps.iloc[ix].longitude_deg) if valid else None)
                handle.write(json.dumps(row, allow_nan=False)+'\n'); count += 1
            if progress:
                progress(min(start+1024, len(x))/len(x))
    summary = dict(session_id=session_id, dataset='user_csv', split='user',
        display_name=display_name or Path(csv_path).name, samples=len(x), duration_s=len(x)/100,
        timestamp_origin_unix_ns=str(origin) if origin is not None else None,
        quality_mode='ordinal', quality_classes=['good','medium','bad'],
        gyro_available=bool(samples[['gyro_x','gyro_y','gyro_z']].notna().any().any()),
        gps_fixes=len(gps), updates=count, incomplete_tail_samples=len(x)%16,
        score_filter_applied=True, alert_filter=DEFAULT_ALERT_FILTER)
    manifest = dict(sessions=[summary], samples=len(x), duration_s=len(x)/100,
        sample_rate_hz=100, quality_mode='ordinal', score_filter_applied=True,
        alert_filter=DEFAULT_ALERT_FILTER, resampling='Causal previous sample; IMU max age 50 ms, speed/GPS 3 s',
        source_csv_sha256=sha(csv_path), input_options=options,
        ensemble_sha256=sha(ensemble) if loaded_from_receipt else None)
    write(output/'manifest.json', manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--session-id', default='uploaded_drive')
    parser.add_argument('--ensemble', type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    parser.add_argument('--time-unit', choices=['s','ms','iso'], default='s')
    parser.add_argument('--acceleration-unit', choices=['m/s2','g'], default='m/s2')
    parser.add_argument('--speed-unit', choices=['m/s','km/h'], default='m/s')
    parser.add_argument('--columns', type=Path, help='JSON mapping canonical names to CSV headers')
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = export_drive(args.csv, args.output, session_id=args.session_id, ensemble=args.ensemble,
        device=args.device, time_unit=args.time_unit, acceleration_unit=args.acceleration_unit,
        speed_unit=args.speed_unit, columns=json.loads(args.columns.read_text()) if args.columns else None)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
