import {Replay, clock, upperBound} from './replay.js';

const $ = id => document.getElementById(id);
const COLORS = ['#159c78', '#d2ae36', '#e78043', '#c95268'];
const NAMES = ['Good', 'Medium', 'Bad', 'Terrible'];
const SENSOR_COLORS = ['#289b7b', '#caaa50', '#748fb7'];
let engine = null, playing = false, position = 0, follow = true, loading = false;
let requestNumber = 0, controller = null, lastFrame = performance.now(), lastPaint = 0;
let rawGpsIndex = -1, routePart = null, routePoints = [], lastCarFix = -1;
let modelRows = [], lastEventSignature = '', catalog = null;
const qualityByFix = new Map(), eventMarkers = new Map();
const params = new URLSearchParams(location.search);

const map = L.map('map', {zoomControl: true, preferCanvas: true, minZoom: 3}).setView([39.618, 22.428], 16);
map.createPane('traveled').style.zIndex = 350;
map.createPane('quality').style.zIndex = 410;
map.createPane('alerts').style.zIndex = 450;
const routeLayer = L.layerGroup().addTo(map);
const qualityLayer = L.layerGroup().addTo(map);
const pendingLayer = L.layerGroup().addTo(map);
const eventLayer = L.layerGroup().addTo(map);
const renderer = L.canvas({padding: .5});
const car = L.marker([0, 0], {zIndexOffset: 1000, interactive: false, icon: L.divIcon({
  className: 'car-icon', iconSize: [42, 42], iconAnchor: [21, 21],
  html: '<div class="car-halo"><div class="car-arrow"><svg viewBox="0 0 24 24"><path d="m12 2 8 19-8-4-8 4Z"/></svg></div></div>',
})});
map.on('dragstart', () => setFollow(false));

function label(session) {
  if (session.dataset === 'user_csv') return session.display_name || session.session_id;
  if (session.dataset === 'kaggle') return 'Kaggle · Larisa, Greece';
  const match = session.session_id.match(/gm_(\d+)_pass_(\d+)/);
  return `LiRA · M13 · Car ${match?.[1]} / Pass ${match?.[2]}`;
}

function qualityText(row) {
  if (!row.valid || row.quality_grade == null) return 'Unknown road quality';
  const name = NAMES[row.quality_grade];
  return row.quality_probability
    ? `${name} · ${(row.quality_probability[row.quality_grade]*100).toFixed(0)}% class probability`
    : `${name} · ${row.iri_m_per_km.toFixed(2)} m/km`;
}

async function refreshCatalog() {
  const response = await fetch('/api/catalog');
  if (!response.ok) throw new Error(`Catalog request failed: HTTP ${response.status}`);
  catalog = await response.json();
  $('upload-panel').hidden = !catalog.uploads_enabled;
  $('session').replaceChildren();
  for (const dataset of ['user_csv', 'kaggle', 'lira_cd']) {
    const group = document.createElement('optgroup');
    group.label = dataset === 'user_csv' ? 'Your drives' : dataset === 'kaggle' ? 'Kaggle Road Quality' : 'LiRA-CD · M13 test road';
    for (const session of catalog.sessions.filter(s => s.dataset === dataset)) {
      const option = document.createElement('option'); option.value = session.session_id;
      option.textContent = label(session); group.append(option);
    }
    if (group.children.length) $('session').append(group);
  }
  $('session').disabled = !catalog.sessions.length;
  $('profile').replaceChildren();
  for (const profile of catalog.profiles) {
    const option = document.createElement('option'); option.value = profile.id; option.textContent = profile.label;
    $('profile').append(option);
  }
  $('profile').disabled = !catalog.sessions.length;
}

$('upload-form').addEventListener('submit', async event => {
  event.preventDefault();
  const file = $('upload-file').files[0];
  if (!file) return;
  setPlaying(false); $('upload-submit').disabled = true;
  const status = $('upload-status');
  try {
    if (file.size > 64*1024*1024) throw new Error('CSV must be at most 64 MiB.');
    const columns = $('column-mapping').value.trim() || '{}';
    JSON.parse(columns);
    status.textContent = 'Uploading recording…';
    const query = new URLSearchParams({filename: file.name, time_unit: $('time-unit').value,
      acceleration_unit: $('acceleration-unit').value, speed_unit: $('speed-unit').value, columns});
    const response = await fetch(`/api/import?${query}`, {method: 'POST', body: file,
      headers: {'Content-Type': 'text/csv'}});
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Upload failed');
    for (;;) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const poll = await fetch(`/api/import/${encodeURIComponent(result.job_id)}`);
      if (!poll.ok) throw new Error('Could not read inference progress');
      const job = await poll.json();
      if (job.status === 'failed') throw new Error(job.error);
      status.textContent = `Running ensemble inference… ${(job.progress*100).toFixed(0)}%`;
      if (job.status === 'complete') {
        await refreshCatalog();
        status.textContent = 'Inference ready. Replaying your recording.';
        await loadSession(job.session_id, 0, true, 'original');
        break;
      }
    }
  } catch (error) {
    status.textContent = `Could not process recording: ${error.message}`;
  } finally {
    $('upload-submit').disabled = false;
  }
});

function setFollow(value) {
  follow = value;
  $('follow').classList.toggle('active', value);
  $('follow').setAttribute('aria-pressed', String(value));
  if (value && engine?.gps.valid) map.panTo([engine.gps.lat, engine.gps.lon], {animate: false});
}

function setPlaying(value) {
  playing = Boolean(value && engine && !loading);
  lastFrame = performance.now();
  $('play').setAttribute('aria-label', playing ? 'Pause' : 'Play');
  $('play').innerHTML = playing
    ? '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zm7 0h4v14h-4z"/></svg>'
    : '<svg viewBox="0 0 24 24"><path d="m9 5 11 7-11 7Z"/></svg>';
  if (engine) renderReadout();
}

function clearMap() {
  routeLayer.clearLayers(); qualityLayer.clearLayers(); pendingLayer.clearLayers(); eventLayer.clearLayers();
  qualityByFix.clear(); eventMarkers.clear();
  if (map.hasLayer(car)) map.removeLayer(car);
  rawGpsIndex = -1; routePart = null; routePoints = []; lastCarFix = -1;
  lastEventSignature = '';
}

async function loadSession(id, initialTime = 0, autoplay = true, profile = $('profile').value || 'original') {
  if (catalog?.sessions.find(s => s.session_id === id)?.dataset === 'user_csv') profile = 'original';
  const request = ++requestNumber;
  controller?.abort(); controller = new AbortController();
  setPlaying(false); loading = true;
  $('loading').hidden = false; $('load-error').hidden = true;
  $('play').disabled = true; $('seek').disabled = true;
  engine = null; clearMap();
  for (const id of ['iri', 'speed', 'gps-state']) $(id).textContent = '—';
  $('quality-badge').textContent = 'Waiting'; $('quality-badge').style.color = '#74886b';
  $('quality-badge').style.background = '#edf4ee';
  $('quality-pointer').style.left = '0%';
  $('probability').innerHTML = '—<small>%</small>'; $('probability-fill').style.width = '0%';
  $('disturbance-state').textContent = 'Waiting'; $('disturbance-card').classList.remove('alert');
  $('disturbance-note').textContent = 'Loading the selected recording';
  $('quality-note').textContent = 'Waiting for the first finalized estimate';
  $('final-status').textContent = 'Loading recording'; $('gps-age').textContent = 'No fix yet';
  $('event-count').textContent = '0'; $('events').replaceChildren();
  $('task-note').textContent = ''; $('playback-label').textContent = 'Loading recording';
  $('elapsed').textContent = '00:00.0'; $('seek').value = '0'; $('utc-clock').textContent = '—';
  for (const id of ['acc-chart', 'gyro-chart', 'model-chart']) {
    const canvas = $(id); canvas.getContext('2d').clearRect(0, 0, canvas.width, canvas.height);
  }
  try {
    const response = await fetch(`/api/session/${encodeURIComponent(id)}?profile=${encodeURIComponent(profile)}`, {signal: controller.signal});
    if (!response.ok) throw new Error(`The recording service returned HTTP ${response.status}.`);
    const data = await response.json();
    if (request !== requestNumber) return;
    engine = new Replay(data); position = 0; loading = false;
    const ordinal = data.session.quality_mode === 'ordinal';
    document.body.classList.toggle('ordinal-quality', ordinal);
    $('terrible-legend').hidden = ordinal;
    $('quality-unit').innerHTML = ordinal ? 'predicted<span class="unit-caption">road class</span>' : 'm/km<span class="unit-caption">estimated IRI</span>';
    $('quality-info').title = ordinal ? 'Good / medium / bad ordinal prediction; not a numeric IRI estimate' : 'Predicted International Roughness Index, metres per kilometre';
    $('quality-line-label').textContent = ordinal ? 'Quality 0–2' : 'IRI';
    $('session').value = id;
    $('profile').value = profile;
    const profileOption = [...$('profile').options].find(o => o.value === profile);
    if (profileOption) profileOption.textContent = data.profile.label;
    $('profile').disabled = data.session.dataset === 'user_csv' || catalog.profiles.length <= 1;
    const kalman = data.profile.config?.kind === 'kalman' || profile === 'kalman';
    $('profile-note').textContent = data.profile.applied_in_export ? data.profile.label : profile === 'original' ? 'Consensus + hysteresis' : profile === 'threshold' ? 'Higher precision · lower recall' : 'Experimental · fewer alerts, more misses';
    $('model-line-note').textContent = kalman ? 'Solid: filtered · dashed: raw provisional' : 'Solid: final · dashed: provisional';
    document.querySelector('.download').textContent = 'Inference data ↓';
    document.querySelector('.download').href = `/api/updates/${encodeURIComponent(id)}?profile=${encodeURIComponent(profile)}`;
    document.querySelector('.download').title = 'Download timestamped predictions and target GPS locations';
    $('session-meta').textContent = `${data.session.samples.toLocaleString()} samples · ${clock(data.session.duration_s)} drive · 100 Hz inputs`;
    $('duration').textContent = clock(data.duration_s);
    $('seek').max = String(data.duration_s);
    $('seek').disabled = false; $('play').disabled = false;
    $('loading').hidden = true;
    $('place').textContent = data.session.dataset === 'user_csv' ? label(data.session) : data.session.dataset === 'kaggle' ? 'Larisa, Greece' : 'M13 · Copenhagen, Denmark';
    const isLira = data.session.dataset === 'lira_cd';
    const missingGyro = isLira || data.session.gyro_available === false;
    $('gyro-missing').hidden = !missingGyro;
    $('gyro-missing').querySelector('span').textContent = 'The model uses acceleration and speed; gyro is optional for display.';
    $('gyro-footer').textContent = missingGyro ? 'Not recorded' : 'Recorded sensor axes';
    $('task-note').textContent = data.session.dataset === 'user_csv'
      ? 'Your recording · model predictions without reference labels. Missing GPS leaves the map empty; sensor inference continues.' : isLira
      ? 'LiRA provides measured roughness. Disturbance predictions have no reference labels here.'
      : 'Kaggle provides disturbance labels. Roughness estimates have no measured IRI reference here.';
    if (data.gps.length) map.setView(data.gps[0].slice(1), 16, {animate: false});
    setFollow(true);
    seekTo(initialTime);
    history.replaceState(null, '', `?drive=${encodeURIComponent(id)}&profile=${encodeURIComponent(profile)}`);
    setPlaying(autoplay && !engine.ended);
  } catch (error) {
    if (error.name === 'AbortError' || request !== requestNumber) return;
    loading = false; $('loading').hidden = true; $('load-error').hidden = false;
    $('error-message').textContent = error.message;
  }
}

function paintQuality(row) {
  if (!row.is_final || !row.valid || !row.target_gps_valid) return;
  const index = row.target_gps_fix_index;
  const existing = qualityByFix.get(index);
  if (existing && existing.patch > row.target_patch) return;
  const fix = engine.data.gps[index], previous = engine.data.gps[index - 1];
  if (!fix) return;
  const options = {color: COLORS[row.quality_grade], weight: 6, opacity: .88, pane: 'quality', renderer};
  const tooltip = `${qualityText(row)}<br>Target ${clock(row.start_s, true)} · final at ${clock(row.available_s, true)}`;
  if (existing) {
    existing.layer.setStyle(options).setTooltipContent(tooltip);
    existing.patch = row.target_patch;
    return;
  }
  const layer = previous && fix[0] - previous[0] <= 3
    ? L.polyline([previous.slice(1), fix.slice(1)], options)
    : L.circleMarker(fix.slice(1), {...options, radius: 3, weight: 1, fillOpacity: .9});
  layer.bindTooltip(tooltip, {sticky: true}).addTo(qualityLayer);
  qualityByFix.set(index, {patch: row.target_patch, layer});
}

function eventPopup(id) {
  const event = engine.events.get(id);
  if (!event) return '';
  const ending = event.closed ? (event.censored ? 'End unknown (missing input)' : `Ended ${clock(event.end, true)}`)
    : engine.ended ? 'End unknown (recording ended)' : 'Ongoing in finalized timeline';
  return `<strong>Disturbance #${id}</strong><br>Road time ${clock(event.start, true)}<br>Alert available ${clock(event.available, true)}<br>${ending}<br><small>Highest score observed: ${(event.probability * 100).toFixed(0)}%</small>`;
}

function syncMap(changed, reset) {
  if (reset) clearMap();
  while (rawGpsIndex < engine.gpsIndex) {
    const index = ++rawGpsIndex;
    const fix = engine.data.gps[index], previous = engine.data.gps[index - 1];
    if (!previous || fix[0] - previous[0] > 3) {
      routePart = L.polyline([], {color: '#a4b1a0', opacity: .8, weight: 2, pane: 'traveled', renderer}).addTo(routeLayer);
    }
    routePart.addLatLng(fix.slice(1)); routePoints.push(fix.slice(1));
  }
  for (const row of changed) {
    paintQuality(row);
    if (row.is_final && row.event_transition === 'start' && row.target_gps_valid && !eventMarkers.has(row.event_id)) {
      const marker = L.circleMarker([row.target_latitude_deg, row.target_longitude_deg], {
        radius: 5, color: '#ce754d', fillColor: '#fff5e9', weight: 2, fillOpacity: 1, pane: 'alerts', renderer,
      }).bindPopup(() => eventPopup(row.event_id)).addTo(eventLayer);
      eventMarkers.set(row.event_id, marker);
    }
  }
  pendingLayer.clearLayers();
  for (const row of engine.provisional) {
    if (!row.valid || !row.target_gps_valid) continue;
    L.circleMarker([row.target_latitude_deg, row.target_longitude_deg], {
      radius: 5, color: '#758b68', fill: false, weight: 1.3, dashArray: '2,3', pane: 'alerts', renderer,
    }).bindTooltip(`Provisional · ${qualityText(row)}<br>No final disturbance decision yet`).addTo(pendingLayer);
  }
  const gps = engine.gps;
  if (!gps.valid) {
    if (map.hasLayer(car)) map.removeLayer(car);
    if (follow && gps.lat != null && lastCarFix !== engine.gpsIndex) {
      // Keep the viewport near the last observed location, with no current car.
      map.panTo([gps.lat, gps.lon], {animate: false}); lastCarFix = engine.gpsIndex;
    }
  } else {
    if (!map.hasLayer(car)) car.addTo(map);
    if (lastCarFix !== engine.gpsIndex) {
      car.setLatLng([gps.lat, gps.lon]);
      const previous = engine.data.gps[engine.gpsIndex - 1];
      if (previous) {
        const heading = Math.atan2((gps.lon - previous[2]) * Math.cos(gps.lat * Math.PI / 180), gps.lat - previous[1]) * 180 / Math.PI;
        car.getElement()?.style.setProperty('--heading', `${heading}deg`);
      }
      if (follow) map.panTo([gps.lat, gps.lon], {animate: false});
      lastCarFix = engine.gpsIndex;
    }
  }
}

function seekTo(time) {
  if (!engine || loading) return;
  position = Math.max(0, Math.min(engine.data.duration_s, Number(time)));
  const result = engine.seek(position);
  syncMap(result.changed, result.reset);
  modelRows = [...engine.visible.values()];
  renderReadout(); drawCharts();
  if (engine.ended) setPlaying(false);
}

function renderReadout() {
  const final = engine.latestFinal, newest = engine.newest, gps = engine.gps;
  $('elapsed').textContent = clock(engine.time, true);
  $('seek').value = String(engine.time);
  const progress = engine.time / engine.data.duration_s * 100;
  $('seek').style.background = `linear-gradient(to right,#689977 ${progress}%,#e6ece3 ${progress}%)`;
  const label = engine.ended ? 'Drive complete · tail remains provisional' : playing ? 'Playing recorded drive' : 'Paused · explore the timeline';
  $('playback-label').textContent = label;
  $('map-status').innerHTML = `<i></i>${engine.ended ? 'DRIVE COMPLETE' : playing ? 'REPLAY IN PROGRESS' : 'REPLAY PAUSED'}`;
  $('coverage').textContent = `${Math.max(0, engine.sampleIndex + 1).toLocaleString()} / ${engine.data.session.samples.toLocaleString()} samples observed`;
  const originValue = engine.data.session.timestamp_origin_unix_ns;
  const origin = Number(originValue) / 1e6;
  $('utc-clock').textContent = originValue == null ? `Relative drive time · ${clock(engine.time, true)}` : new Date(origin + engine.time * 1000).toISOString().replace('T', ' ').slice(0, 23) + ' UTC';
  const ordinal = engine.data.session.quality_mode === 'ordinal';
  const grade = final?.valid ? final.quality_grade : null;
  const iri = final?.valid ? final.iri_m_per_km : null;
  $('iri').textContent = ordinal ? grade == null ? '—' : NAMES[grade] : iri == null ? '—' : iri.toFixed(2);
  $('quality-badge').textContent = grade == null ? 'Waiting' : ordinal ? `${(final.quality_probability[grade]*100).toFixed(0)}%` : NAMES[grade];
  $('quality-badge').style.color = grade == null ? '#74886b' : COLORS[grade];
  $('quality-badge').style.background = grade == null ? '#edf4ee' : COLORS[grade] + '14';
  $('quality-pointer').style.left = `${ordinal ? grade == null ? 0 : (grade+.5)/3*100 : iri == null ? 0 : Math.min(99, iri / 8 * 100)}%`;
  $('quality-note').textContent = final
    ? `Final for road time ${clock(final.start_s, true)}–${clock(final.end_s, true)}`
    : 'Waiting for the first finalized estimate';
  $('final-status').textContent = engine.ended ? `${engine.provisional.length} provisional tail patches` : final ? 'Final estimates' : 'Collecting context';
  const probability = final?.valid ? final.probability : null;
  $('probability').innerHTML = `${probability == null ? '—' : (probability * 100).toFixed(0)}<small>%</small>`;
  $('probability-fill').style.width = `${(probability || 0) * 100}%`;
  $('disturbance-state').textContent = probability == null ? 'Waiting' : final.disturbance ? 'Disturbance' : 'No disturbance';
  $('disturbance-card').classList.toggle('alert', Boolean(final?.disturbance));
  $('disturbance-note').textContent = final?.disturbance ? `Event #${final.event_id} · combined defect category` : 'Manholes, cracks, bumps & depressions';
  const speed = engine.sensor('speed');
  $('speed').textContent = speed == null ? '—' : (speed * 3.6).toFixed(0);
  $('gps-state').textContent = gps.valid ? 'Observed' : gps.age == null ? 'Waiting' : 'GPS gap';
  $('gps-age').textContent = gps.age == null ? 'No fix yet' : `Last fix ${gps.age.toFixed(1)} s ago`;
  $('gps-dot').style.background = gps.valid ? '#7aa67f' : '#d0a568';
  $('map-caption').textContent = !gps.valid
    ? gps.age == null ? 'Waiting for the first GPS fix.' : 'GPS is stale. Sensor readings and predictions continue.'
    : `Observed GPS · ${gps.lat.toFixed(5)}, ${gps.lon.toFixed(5)} · hollow points are provisional`;
  const events = [...engine.events.values()].reverse();
  $('event-count').textContent = String(events.length);
  const signature = events.slice(0, 10).map(e => `${e.id}:${e.closed}:${e.censored}`).join(',') + engine.ended;
  if (signature !== lastEventSignature) {
    lastEventSignature = signature;
    $('events').replaceChildren();
    if (!events.length) {
      const empty = document.createElement('div'); empty.className = 'empty-events';
      empty.textContent = 'New disturbance alerts will appear here as you drive.'; $('events').append(empty);
    }
    for (const event of events.slice(0, 12)) {
      const button = document.createElement('button'); button.className = 'event-item';
      const state = event.closed ? (event.censored ? 'End unknown' : `${(event.end - event.start).toFixed(2)} s`) : engine.ended ? 'End unknown' : 'Ongoing';
      button.innerHTML = `<span class="event-dot"></span><span class="event-text">Disturbance #${event.id}<small>${event.gps ? 'Located on route' : 'GPS unavailable'} · ${state}</small></span><time>${clock(event.start, true)}</time>`;
      button.addEventListener('click', () => {
        setPlaying(false); seekTo(event.available);
        if (event.gps) {setFollow(false); map.setView([event.lat, event.lon], Math.max(16, map.getZoom())); eventMarkers.get(event.id)?.openPopup();}
      });
      $('events').append(button);
    }
  }
}

function canvasSetup(canvas, min, max, rightLabel = null) {
  const rect = canvas.getBoundingClientRect(), ratio = Math.min(2, window.devicePixelRatio || 1);
  const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);
  if (canvas.width !== width || canvas.height !== height) {canvas.width = width; canvas.height = height;}
  const ctx = canvas.getContext('2d'); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height);
  const plot = {x: 31, y: 9, w: Math.max(1, rect.width - (rightLabel ? 66 : 41)), h: rect.height - 28};
  ctx.font = '8px system-ui'; ctx.lineWidth = .6; ctx.strokeStyle = '#edf0e9'; ctx.fillStyle = '#a3af9f';
  for (let i = 0; i < 3; i++) {
    const y = plot.y + plot.h * i / 2;
    ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText((max - (max - min) * i / 2).toFixed(max - min < 2 ? 2 : 0), plot.x - 5, y + 3);
    if (rightLabel) {ctx.textAlign = 'left'; ctx.fillText(`${100 - i * 50}%`, plot.x + plot.w + 5, y + 3);}
  }
  const end = engine.time, start = Math.max(0, end - 12);
  const left = Math.max(0, end - 12), right = Math.max(12, end);
  ctx.textAlign = 'left'; ctx.fillText(`${left.toFixed(0)} s`, plot.x, rect.height - 3);
  ctx.textAlign = 'right'; ctx.fillText(`${right.toFixed(0)} s`, plot.x + plot.w, rect.height - 3);
  return {ctx, plot, start, end, left, right, min, max};
}

function drawLine(chart, points, color, min = chart.min, max = chart.max, dashed = false) {
  const {ctx, plot, left, right} = chart;
  ctx.save(); ctx.beginPath(); ctx.rect(plot.x, plot.y - 2, plot.w, plot.h + 4); ctx.clip();
  ctx.strokeStyle = color; ctx.lineWidth = 1.1; ctx.setLineDash(dashed ? [3, 3] : []); ctx.beginPath();
  let active = false;
  for (const [time, value] of points) {
    if (value == null || !Number.isFinite(value)) {active = false; continue;}
    const x = plot.x + (time - left) / (right - left) * plot.w;
    const y = plot.y + (1 - (value - min) / (max - min)) * plot.h;
    if (active) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    active = true;
  }
  ctx.stroke(); ctx.restore();
}

function shadeEvents(chart) {
  const {ctx, plot, left, right} = chart; ctx.fillStyle = '#d18c6010';
  for (const event of engine.events.values()) {
    const end = Math.min(event.until, engine.time), start = Math.max(event.start, left);
    if (end <= left || start >= right) continue;
    ctx.fillRect(plot.x + (start - left) / (right - left) * plot.w, plot.y,
      Math.max(1, (end - start) / (right - left) * plot.w), plot.h);
  }
}

function drawSensors(id, names) {
  const data = engine.data.signals.data, c = engine.columns;
  const from = upperBound(data, Math.max(0, engine.time - 12) - .01, row => row[c.time_s]);
  const until = engine.sampleIndex + 1;
  const series = names.map(name => {
    const points = [];
    for (let i = from; i < until; i++) points.push([data[i][c.time_s], data[i][c[name]]]);
    return points;
  });
  const values = series.flatMap(points => points.map(p => p[1]).filter(v => v != null));
  let low = values.length ? Math.min(...values) : -1, high = values.length ? Math.max(...values) : 1;
  const pad = Math.max((high - low) * .12, id === 'gyro-chart' ? .02 : .5);
  const chart = canvasSetup($(id), low - pad, high + pad);
  shadeEvents(chart);
  series.forEach((points, i) => drawLine(chart, points, SENSOR_COLORS[i]));
}

function drawCharts() {
  if (!engine) return;
  drawSensors('acc-chart', ['accel_x', 'accel_y', 'accel_z']);
  drawSensors('gyro-chart', ['gyro_x', 'gyro_y', 'gyro_z']);
  const rows = modelRows.filter(row => row.end_s >= engine.time - 12);
  const ordinal = engine.data.session.quality_mode === 'ordinal';
  const qualityMax = ordinal ? 2 : 8;
  const chart = canvasSetup($('model-chart'), 0, qualityMax, true);
  shadeEvents(chart);
  for (const provisional of [false, true]) {
    const use = rows.filter(row => row.is_final !== provisional).sort((a, b) => a.start_s - b.start_s);
    // Duplicate interval endpoints so patch estimates remain steps, not ramps.
    const iri = [], probability = [];
    for (const row of use) {
      for (const time of [row.start_s, Math.min(row.end_s, engine.time)]) {
        iri.push([time, row.valid ? ordinal ? row.quality_grade : row.iri_m_per_km : null]);
        probability.push([time, row.valid ? row.probability : null]);
      }
    }
    drawLine(chart, iri, '#289b7b', 0, qualityMax, provisional);
    drawLine(chart, probability, '#d08061', 0, 1, provisional);
  }
}

$('session').addEventListener('change', () => loadSession($('session').value));
$('profile').addEventListener('change', () => {
  const time = engine?.time || 0, resume = playing;
  loadSession($('session').value, time, resume, $('profile').value);
});
$('play').addEventListener('click', () => {if (engine?.ended) seekTo(0); setPlaying(!playing);});
$('restart').addEventListener('click', () => {seekTo(0); setPlaying(false);});
$('next-event').addEventListener('click', () => {if (engine) {setPlaying(false); seekTo(engine.nextAlert());}});
$('seek').addEventListener('input', event => {
  const target = Number(event.target.value); // Pausing redraws the slider.
  setPlaying(false); seekTo(target);
});
$('follow').addEventListener('click', () => setFollow(!follow));
$('fit').addEventListener('click', () => {
  if (!routePoints.length) return;
  setFollow(false); map.fitBounds(L.latLngBounds(routePoints), {padding: [50, 65], maxZoom: 17});
});
$('retry').addEventListener('click', () => catalog ? loadSession($('session').value) : location.reload());
for (const id of ['about', 'method-button']) $(id).addEventListener('click', () => $('about-dialog').showModal());
$('close-about').addEventListener('click', () => $('about-dialog').close());
$('about-dialog').addEventListener('click', event => {if (event.target === $('about-dialog')) {
  const rect = event.target.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close();
}});
document.addEventListener('keydown', event => {
  if (/^(INPUT|SELECT|TEXTAREA|BUTTON)$/.test(event.target.tagName) || $('about-dialog').open) return;
  if (event.code === 'Space') {event.preventDefault(); if (engine?.ended) seekTo(0); setPlaying(!playing);}
  if (event.code === 'ArrowRight' || event.code === 'ArrowLeft') {
    event.preventDefault(); setPlaying(false); seekTo(position + (event.code === 'ArrowRight' ? 5 : -5));
  }
});
document.addEventListener('visibilitychange', () => {if (document.hidden) setPlaying(false);});
new ResizeObserver(() => {map.invalidateSize(); if (engine) drawCharts();}).observe(document.querySelector('.workspace'));
window.addEventListener('resize', drawCharts);

function frame(now) {
  const dt = (now - lastFrame) / 1000; lastFrame = now;
  if (playing && engine) {
    position = Math.min(engine.data.duration_s, position + dt * Number($('rate').value));
    if (now - lastPaint >= 75 || position >= engine.data.duration_s) {seekTo(position); lastPaint = now;}
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// A read-only diagnostic used by browser tests; includes only currently visible state.
window.replaySnapshot = () => engine ? {
  session: engine.data.session.session_id, time: engine.time, duration: engine.data.duration_s,
  profile: engine.data.profile?.id || 'original',
  sampleIndex: engine.sampleIndex, gps: engine.gps, visibleUpdates: engine.cursor,
  finalCount: engine.finalCount, provisionalCount: engine.provisional.length,
  maxVisibleAvailability: Math.max(0, ...[...engine.visible.values()].map(u => u.available_s)),
  events: engine.events.size, mapEvents: eventMarkers.size, coloredFixes: qualityByFix.size,
  playing, carVisible: map.hasLayer(car), latestFinal: engine.latestFinal,
} : null;

try {
  await refreshCatalog();
  const tiles = L.tileLayer(catalog.tile_url, {maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors', keepBuffer: 2}).addTo(map);
  let failed = 0;
  tiles.on('tileerror', () => {failed++; if (failed >= 3) $('tile-warning').hidden = false;});
  tiles.on('tileload', () => {failed = 0; $('tile-warning').hidden = true;});
  if (catalog.sessions.length) {
    const id = catalog.sessions.some(s => s.session_id === params.get('drive')) ? params.get('drive') : catalog.sessions[0].session_id;
    const profile = catalog.profiles.some(p => p.id === params.get('profile')) ? params.get('profile') : catalog.default_profile;
    await loadSession(id, Number(params.get('t')) || 0, params.get('autoplay') !== '0', profile);
  } else {
    $('loading').hidden = true;
    $('map-status').textContent = 'UPLOAD A DRIVE';
    $('place').textContent = 'Your recording';
    $('session-meta').textContent = 'No recordings yet';
    $('map-caption').textContent = 'Upload a CSV above to run inference and replay your drive.';
  }
} catch (error) {
  $('loading').hidden = true; $('load-error').hidden = false; $('error-message').textContent = error.message;
}
