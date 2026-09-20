"""Apply a fixed car calibration to already aggregated ordinal predictions."""
import hashlib
import json
import math

NAMES = ('good', 'medium', 'bad')


def load_calibrations(path, sessions, folders):
    """Bind a calibration to exact recordings, never to arbitrary new uploads."""
    if path is None:
        return {}
    saved = json.loads(path.read_text())
    offset = float(saved['offset'])
    if saved.get('method') != 'ordinal_logit_offset' or not math.isfinite(offset) or not 0 <= offset <= 6:
        raise ValueError('Expected an ordinal logit offset between 0 and 6')
    result = {}
    for session_id, digest in saved['updates_sha256'].items():
        if session_id not in sessions or sessions[session_id].get('quality_mode') != 'ordinal':
            raise ValueError(f'Calibration requires ordinal recording {session_id}')
        actual = hashlib.sha256((folders[session_id] / 'updates.jsonl.gz').read_bytes()).hexdigest()
        if actual != digest:
            raise ValueError(f'Calibration recording hash differs: {session_id}')
        result[session_id] = dict(id=saved['id'], label=saved['label'], offset=offset,
                                  reference=saved['reference'])
    return result


def calibrated_update(row, calibration):
    """Only quality fields change; times, GPS, defect scores and events survive."""
    probability = row.get('quality_probability')
    if not row.get('valid') or probability is None:
        return row
    if (len(probability) != 3 or any(not math.isfinite(p) or p < 0 for p in probability)
            or not math.isclose(sum(probability), 1., abs_tol=1e-5)):
        raise ValueError('Expected three normalized ordinal probabilities')
    # Equivalent to sigmoid(logit(c) - offset), without exp overflow.
    multiplier = math.exp(calibration['offset'])
    def shift(c):
        c = min(1 - 1e-7, max(1e-7, c))
        return c / (c + (1 - c) * multiplier)
    above_good, bad = shift(sum(probability[1:])), shift(probability[2])
    calibrated = [1 - above_good, above_good - bad, bad]
    grade = int(sum(calibrated[1:]) >= .5) + int(bad > .5)
    return dict(row, quality_probability=calibrated, quality_grade=grade,
                quality_name=NAMES[grade], quality_calibration_id=calibration['id'],
                original_quality_probability=probability,
                original_quality_grade=row.get('quality_grade'),
                original_quality_name=row.get('quality_name'))
