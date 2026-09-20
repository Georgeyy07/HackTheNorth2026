import test from 'node:test';
import assert from 'node:assert/strict';

import { buildQualitySegments, QUALITY_COLORS, scoreColor, severityWeight } from '../src/routeQuality.js';

// A straight north-bound run of 31 points, ~0.0001 degrees apart.
const line = Array.from({ length: 31 }, (_, i) => [43.47 + i * 0.0001, -80.54]);

test('a route with no reported potholes is green end to end', () => {
  const segments = buildQualitySegments(line, []);
  assert.ok(segments.length > 1);
  assert.ok(segments.every((s) => s.color === QUALITY_COLORS.clean));
});

test('only the stretch near a pothole changes colour', () => {
  // Sits on the first segment (indices 0-9).
  const segments = buildQualitySegments(line, [{ lat: 43.4702, lon: -80.54, severity: 'MEDIUM' }]);
  assert.equal(segments[0].color, QUALITY_COLORS.moderate);
  assert.ok(segments.slice(1).every((s) => s.color === QUALITY_COLORS.clean));
});

test('severity escalates a stretch from moderate to rough', () => {
  const mild = buildQualitySegments(line, [{ lat: 43.4702, lon: -80.54, severity: 'LOW' }]);
  assert.equal(mild[0].color, QUALITY_COLORS.moderate);

  const severe = buildQualitySegments(line, [{ lat: 43.4702, lon: -80.54, severity: 'CRITICAL' }]);
  assert.equal(severe[0].color, QUALITY_COLORS.rough);
});

test('several mild potholes on one stretch add up to rough', () => {
  const potholes = [
    { lat: 43.4701, lon: -80.54, severity: 'LOW' },
    { lat: 43.4702, lon: -80.54, severity: 'LOW' },
    { lat: 43.4703, lon: -80.54, severity: 'LOW' },
  ];
  assert.equal(buildQualitySegments(line, potholes)[0].color, QUALITY_COLORS.rough);
});

test('segments join end to end so the drawn line has no gaps', () => {
  const segments = buildQualitySegments(line, []);
  for (let i = 1; i < segments.length; i++) {
    const previousEnd = segments[i - 1].coordinates.at(-1);
    const currentStart = segments[i].coordinates[0];
    assert.deepEqual(currentStart, previousEnd);
  }
  // Together they still cover the whole route, start to finish.
  assert.deepEqual(segments[0].coordinates[0], { latitude: line[0][0], longitude: line[0][1] });
  assert.deepEqual(segments.at(-1).coordinates.at(-1), { latitude: line.at(-1)[0], longitude: line.at(-1)[1] });
});

test('a pothole at the very end lands on the last segment, not past it', () => {
  const end = line.at(-1);
  const segments = buildQualitySegments(line, [{ lat: end[0], lon: end[1], severity: 'CRITICAL' }]);
  assert.equal(segments.at(-1).color, QUALITY_COLORS.rough);
});

test('degenerate routes and unusable coordinates are ignored, not crashed on', () => {
  assert.deepEqual(buildQualitySegments([], []), []);
  assert.deepEqual(buildQualitySegments([[43.47, -80.54]], []), []);
  const segments = buildQualitySegments(line, [{ lat: null, lon: undefined, severity: 'HIGH' }]);
  assert.ok(segments.every((s) => s.color === QUALITY_COLORS.clean));
});

test('an unrecognised severity still counts rather than being dropped', () => {
  // The severity column has been renamed before; an unknown label must not
  // silently render a damaged road as clean.
  assert.equal(severityWeight('SOMETHING_NEW'), 1);
  assert.equal(scoreColor(0), QUALITY_COLORS.clean);
});
