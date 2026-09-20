import test from 'node:test';
import assert from 'node:assert/strict';
import {Replay, upperBound} from '../static/replay.js';

function fixture() {
  const row = (target, available, final, extra = {}) => ({target_patch: target, start_s: target * .16,
    end_s: (target + 1) * .16, available_s: available, is_final: final, valid: true,
    probability: .1, iri_m_per_km: 1., event_id: null, disturbance: final ? false : null,
    ...extra});
  return {duration_s: 5, gps_max_age_s: 3, signals: {columns: ['time_s', 'input_available_s', 'accel_x', 'speed'],
    data: Array.from({length: 500}, (_, i) => [i / 100, i > 0 && i < 4 ? .035 : i / 100, i, 10])},
    gps: [[.12, 39.6, 22.4], [4.2, 39.7, 22.5]], updates: [
      row(0, .16, false), row(0, .32, false, {probability: .4}), row(1, .32, false),
      row(0, .48, true, {probability: .8, disturbance: true, event_id: 1, event_transition: 'start'}),
      row(1, .48, false), row(2, .48, false),
      row(1, .64, true, {event_id: 1, event_transition: 'end'}), row(2, .64, false), row(3, .64, false),
      row(2, .8, true, {probability: .9, disturbance: true, event_id: 2, event_transition: 'start'}),
      row(3, .8, false), row(4, .8, false),
    ]};
}

test('scores, final labels, and events are gated by arrival, including revisions', () => {
  const replay = new Replay(fixture());
  replay.seek(.15); assert.equal(replay.visible.size, 0);
  replay.seek(.16); assert.equal(replay.visible.get(0).probability, .1);
  assert.equal(replay.latestFinal, null); assert.equal(replay.events.size, 0);
  replay.seek(.32); assert.equal(replay.visible.get(0).probability, .4);
  replay.seek(.47999); assert.equal(replay.events.size, 0);
  replay.seek(.48); assert.equal(replay.finalCount, 1);
  assert.equal(replay.events.size, 1); assert.equal(replay.events.get(1).end, null);
  assert.equal(replay.events.get(1).closed, false);
  assert.equal(replay.provisional.length, 2);
  replay.seek(.64); assert.equal(replay.events.get(1).end, .16);
  assert.equal(replay.events.get(1).probability, .8);
  assert.equal(replay.events.size, 1);
});

test('rewind clears future colors, labels, GPS, and event state', () => {
  const replay = new Replay(fixture());
  replay.seek(5); assert.equal(replay.events.size, 2);
  const result = replay.seek(.1);
  assert.equal(result.reset, true); assert.equal(replay.events.size, 0);
  assert.equal(replay.visible.size, 0); assert.equal(replay.finalCount, 0);
  assert.equal(replay.latestFinal, null); assert.equal(replay.gps.valid, false);
  replay.seek(.48); assert.equal(replay.events.get(1).closed, false);
});

test('GPS waits for first fix and expires without borrowing a future fix', () => {
  const replay = new Replay(fixture());
  replay.seek(.119); assert.equal(replay.gps.valid, false);
  replay.seek(.12); assert.equal(replay.gps.lat, 39.6);
  replay.seek(3.121); assert.equal(replay.gps.valid, false);
  replay.seek(4.19); assert.equal(replay.gps.valid, false);
  replay.seek(4.2); assert.equal(replay.gps.valid, true);
  assert.equal(replay.gps.lat, 39.7);
});

test('interpolated sensors wait for their actual availability', () => {
  const replay = new Replay(fixture());
  replay.seek(.03); assert.equal(replay.sampleIndex, 0);
  assert.equal(replay.sensor('accel_x'), 0);
  replay.seek(.035); assert.equal(replay.sampleIndex, 3);
  replay.seek(.04); assert.equal(replay.sampleIndex, 4);
});

test('jump and incremental playback produce the same visible state', () => {
  const a = new Replay(fixture()), b = new Replay(fixture());
  for (let t = 0; t < 5; t += .013) a.seek(t);
  a.seek(5); b.seek(5);
  assert.deepEqual([...a.visible], [...b.visible]);
  assert.deepEqual([...a.events], [...b.events]);
  assert.equal(a.finalCount, 3); assert.equal(a.provisional.length, 2);
  assert.equal(a.events.get(2).end, null);
  b.seek(-1); assert.equal(b.time, 0);
  assert.equal(b.nextAlert(), .48);
  assert.equal(upperBound([.1, .2, .2, .3], .2), 3);
});
