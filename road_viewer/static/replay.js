// A clock-driven reducer. Every visible prediction comes from an arrived update.
export function upperBound(values, time, get = x => x) {
  let lo = 0, hi = values.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (get(values[mid]) <= time) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

export class Replay {
  constructor(data) {
    this.data = data;
    this.columns = Object.fromEntries(data.signals.columns.map((name, index) => [name, index]));
    this.alerts = data.updates.filter(u => u.is_final && u.event_transition === 'start');
    this.reset();
  }
  reset() {
    this.time = 0;
    this.cursor = 0;
    this.visible = new Map();
    this.events = new Map();
    this.finalCount = 0;
    this.latestFinal = null;
    this.newest = null;
    this.sampleIndex = -1;
    this.gpsIndex = -1;
  }
  seek(time) {
    const target = Math.max(0, Math.min(this.data.duration_s, time));
    const reset = target < this.time;
    if (reset) this.reset();
    this.time = target;
    const changed = [];
    while (this.cursor < this.data.updates.length && this.data.updates[this.cursor].available_s <= target) {
      const row = this.data.updates[this.cursor++];
      const previous = this.visible.get(row.target_patch);
      this.visible.set(row.target_patch, row);
      this.newest = !this.newest || row.target_patch >= this.newest.target_patch ? row : this.newest;
      changed.push(row);
      if (!row.is_final) continue;
      if (!previous?.is_final) this.finalCount++;
      this.latestFinal = row;
      if (row.event_transition === 'start') {
        this.events.set(row.event_id, {id: row.event_id, start: row.start_s, available: row.available_s,
          until: row.end_s, end: null, closed: false, censored: false, probability: row.probability,
          lat: row.target_latitude_deg, lon: row.target_longitude_deg, gps: row.target_gps_valid});
      }
      const event = this.events.get(row.event_id);
      if (!event) continue;
      if (row.disturbance) {
        event.until = row.end_s;
        event.probability = Math.max(event.probability, row.probability);
      }
      if (row.event_transition === 'end' || row.event_transition === 'censored') {
        event.closed = true;
        event.censored = row.event_transition === 'censored';
        event.end = event.censored ? null : row.start_s;
      }
    }
    this.sampleIndex = upperBound(this.data.signals.data, target, row => row[this.columns.input_available_s]) - 1;
    this.gpsIndex = upperBound(this.data.gps, target, row => row[0]) - 1;
    return {reset, changed};
  }
  get sample() { return this.data.signals.data[this.sampleIndex] || null; }
  sensor(name) { return this.sample?.[this.columns[name]] ?? null; }
  get gps() {
    const fix = this.data.gps[this.gpsIndex];
    if (!fix) return {valid: false, age: null};
    const age = this.time - fix[0];
    return {lat: fix[1], lon: fix[2], age, valid: age <= this.data.gps_max_age_s};
  }
  get provisional() { return [...this.visible.values()].filter(u => !u.is_final); }
  get ended() { return this.time >= this.data.duration_s; }
  nextAlert() { return this.alerts.find(u => u.available_s > this.time + .001)?.available_s ?? this.data.duration_s; }
}

export function clock(seconds, decimal = false) {
  const num = Number(seconds);
  const value = (!Number.isFinite(num) || num < 0) ? 0 : num;
  const minutes = Math.floor(value / 60);
  const rest = Math.floor(value % 60).toString().padStart(2, '0');
  return `${minutes.toString().padStart(2, '0')}:${rest}${decimal ? '.' + Math.floor((value % 1) * 10) : ''}`;
}
