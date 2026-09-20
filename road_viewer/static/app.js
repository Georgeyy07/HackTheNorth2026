import {ordinalSession, qualityValue, qualityText, visionFrameAt} from './predictions.js';
import {Replay, clock, upperBound} from './replay.js';

const $ = id => document.getElementById(id);
const COLORS = ['#159c78', '#d2ae36', '#e78043', '#c95268'];
const NAMES = ['Good', 'Medium', 'Bad', 'Terrible'];
let ordinal = false;
let visionImageKey = null;
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
map.createPane('potholes').style.zIndex = 500;
map.createPane('routeFinder').style.zIndex = 460;
const routeLayer = L.layerGroup().addTo(map);
const qualityLayer = L.layerGroup().addTo(map);
const pendingLayer = L.layerGroup().addTo(map);
const eventLayer = L.layerGroup().addTo(map);
const potholeLayer = L.layerGroup().addTo(map);
const routeFinderLayer = L.layerGroup().addTo(map);
const renderer = L.canvas({padding: .5});
new ResizeObserver(() => map.invalidateSize({pan: false})).observe($('map'));

let tigerPotholes = [];
const potholeMarkers = new Map();

async function fetchPotholes() {
  try {
    const res = await fetch('/api/potholes');
    if (!res.ok) return;
    tigerPotholes = await res.json();
    renderPotholes();
  } catch (e) {
    console.error("Failed to fetch Tiger Data potholes:", e);
  }
}

function renderPotholes() {
  potholeLayer.clearLayers();
  potholeMarkers.clear();
  
  const countBadge = $('pothole-count-badge');
  if (countBadge) countBadge.textContent = `${tigerPotholes.length} Potholes`;

  const listContainer = $('potholes-list');
  if (listContainer) {
    listContainer.replaceChildren();
    if (!tigerPotholes.length) {
      const empty = document.createElement('div');
      empty.className = 'empty-potholes';
      empty.style.color = '#789c8b';
      empty.style.fontSize = '10px';
      empty.textContent = 'No potholes registered in Tiger Data DB.';
      listContainer.append(empty);
    }
  }

  tigerPotholes.forEach(ph => {
    const markerIcon = L.divIcon({
      className: 'pothole-marker',
      iconSize: [28, 28],
      iconAnchor: [14, 14],
      html: `<div class="pothole-marker-icon" title="${ph.severity} Pothole"><div class="pothole-marker-inner"></div></div>`
    });

    let timeStr = '';
    if (ph.timestamp) {
      const d = new Date(ph.timestamp);
      timeStr = !Number.isNaN(d.getTime()) ? d.toLocaleTimeString() : '';
    }

    const popupContent = `
      <div style="font-family:sans-serif;padding:4px">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:4px">
          <strong style="color:#ff4d6d">⚠️ ${ph.severity} POTHOLE</strong>
          <span style="font-size:9px;background:#0d2b22;color:#00f2fe;padding:2px 5px;border-radius:4px">Tiger DB #${ph.id}</span>
        </div>
        <div style="font-size:9px;color:#666">
          📍 ${ph.latitude.toFixed(4)}, ${ph.longitude.toFixed(4)} ${timeStr ? '• ' + timeStr : ''}
        </div>
        <div style="margin-top:8px;display:flex;gap:6px">
          <button onclick="window.deletePotholeFromDB(${ph.id})" style="background:#ff4d6d;color:#fff;border:none;padding:4px 8px;border-radius:4px;font-size:9px;cursor:pointer">Delete Record</button>
        </div>
      </div>
    `;

    const marker = L.marker([ph.latitude, ph.longitude], {
      icon: markerIcon,
      pane: 'potholes'
    }).bindPopup(popupContent).addTo(potholeLayer);

    potholeMarkers.set(ph.id, marker);

    if (listContainer) {
      const item = document.createElement('div');
      item.className = 'pothole-item';
      item.innerHTML = `
        <div class="pothole-item-header">
          <span class="pothole-sev-badge ${ph.severity}">${ph.severity}</span>
          <span class="pothole-coords">${ph.latitude.toFixed(4)}, ${ph.longitude.toFixed(4)}</span>
        </div>
        <div class="pothole-footer">
          <span>Tiger DB #${ph.id}</span>
          <div class="pothole-footer-btns">
            <button class="tiger-btn danger" onclick="window.deletePotholeFromDB(${ph.id})">✕ Delete</button>
          </div>
        </div>
      `;
      item.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') {
          map.setView([ph.latitude, ph.longitude], 17);
          marker.openPopup();
        }
      });
      listContainer.append(item);
    }
  });
}

window.deletePotholeFromDB = async (id) => {
  await fetch(`/api/potholes/${id}`, {method: 'DELETE'});
  fetchPotholes();
};

window.seedTigerPotholes = async () => {
  await fetch('/api/potholes/seed', {method: 'POST'});
  fetchPotholes();
};

map.on('click', (e) => {
  const form = $('pothole-form');
  if (form) {
    $('ph-lat').value = e.latlng.lat.toFixed(5);
    $('ph-lng').value = e.latlng.lng.toFixed(5);
    form.hidden = false;
  }
});

$('toggle-add-pothole-btn')?.addEventListener('click', () => {
  const form = $('pothole-form');
  if (form) form.hidden = !form.hidden;
});

$('cancel-pothole-btn')?.addEventListener('click', () => {
  const form = $('pothole-form');
  if (form) form.hidden = true;
});

$('seed-potholes-btn')?.addEventListener('click', window.seedTigerPotholes);

$('pothole-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    latitude: parseFloat($('ph-lat').value),
    longitude: parseFloat($('ph-lng').value),
    severity: $('ph-severity').value
  };
  await fetch('/api/potholes', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(payload)
  });
  $('pothole-form').hidden = true;
  fetchPotholes();
});

const ROUTE_COLORS = ['#ff4d6d', '#00f2fe', '#ffb454', '#a78bfa'];
const RISK_COLORS = {None: '#159c78', LOW: '#caaa50', MEDIUM: '#e78043', HIGH: '#c95268', CRITICAL: '#8b1e3f'};
let routePolylines = [];

function selectRoute(index) {
  routePolylines.forEach((line, i) => {
    line.setStyle(i === index ? {weight: 6, opacity: 1} : {weight: 3, opacity: 0.35});
  });
  document.querySelectorAll('#route-results .route-option').forEach((el, i) => {
    el.classList.toggle('selected', i === index);
  });
  if (routePolylines[index]) map.fitBounds(routePolylines[index].getBounds(), {padding: [40, 40]});
}

function formatRoute(route, index, color) {
  const mins = Math.floor(route.duration_s / 60), secs = Math.round(route.duration_s % 60);
  const riskColor = RISK_COLORS[route.risk_rating] || '#789c8b';
  const potholeList = route.potholes_encountered.length
    ? route.potholes_encountered.map(p => `${p.severity} #${p.id}`).join(', ')
    : 'none on this route';
  return `<div class="pothole-item route-option" data-index="${index}" style="cursor:pointer;border-left:3px solid ${color}">` +
    `<div class="pothole-item-header"><span class="pothole-coords">${route.label}</span>` +
    `<span class="pothole-sev-badge" style="background:${riskColor}22;color:${riskColor};border:1px solid ${riskColor}">${route.risk_rating} risk</span></div>` +
    `<div class="pothole-footer"><span>${route.distance_m.toFixed(0)}m · ${mins}m ${secs}s</span></div>` +
    `<div class="pothole-footer"><span>${route.pothole_count} pothole${route.pothole_count === 1 ? '' : 's'}: ${potholeList}</span></div></div>`;
}

function setupAddressAutocomplete(inputId, suggestionsId) {
  const input = $(inputId), box = $(suggestionsId);
  if (!input || !box) return;
  let debounceTimer = null, requestId = 0;

  input.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    const query = input.value.trim();
    if (query.length < 3) {
      box.hidden = true;
      box.replaceChildren();
      return;
    }
    debounceTimer = setTimeout(async () => {
      const thisRequest = ++requestId;
      try {
        const res = await fetch(`/api/geocode/suggest?q=${encodeURIComponent(query)}`);
        const results = await res.json();
        if (thisRequest !== requestId) return;
        box.replaceChildren();
        if (!results.length) {
          box.hidden = true;
          return;
        }
        results.forEach(r => {
          const item = document.createElement('div');
          item.className = 'address-suggestion';
          item.textContent = r.display_name;
          item.addEventListener('click', () => {
            input.value = r.display_name;
            box.hidden = true;
            box.replaceChildren();
          });
          box.append(item);
        });
        box.hidden = false;
      } catch {
        box.hidden = true;
      }
    }, 400);
  });

  input.addEventListener('blur', () => {
    // Delay so a click on a suggestion registers before the dropdown hides.
    setTimeout(() => { box.hidden = true; }, 150);
  });
}

setupAddressAutocomplete('route-origin', 'route-origin-suggestions');
setupAddressAutocomplete('route-destination', 'route-destination-suggestions');

$('route-form')?.addEventListener('submit', async (e) => {
  e.preventDefault();
  const origin = $('route-origin').value.trim();
  const destination = $('route-destination').value.trim();
  const statusEl = $('route-status'), resultsEl = $('route-results');
  resultsEl.replaceChildren();
  statusEl.textContent = 'Finding routes… (first run downloads the road network, ~15-20s)';
  routeFinderLayer.clearLayers();
  routePolylines = [];

  try {
    const url = `/api/route?origin=${encodeURIComponent(origin)}&destination=${encodeURIComponent(destination)}`;
    const res = await fetch(url);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${res.status})`);
    }
    const data = await res.json();

    data.routes.forEach((route, i) => {
      const color = ROUTE_COLORS[i % ROUTE_COLORS.length];
      const line = L.polyline(route.coords, {color, weight: 3, opacity: 0.35, pane: 'routeFinder'})
        .bindTooltip(`${route.label}: ${route.distance_m.toFixed(0)}m, risk ${route.risk_rating}`)
        .on('click', () => selectRoute(i))
        .addTo(routeFinderLayer);
      routePolylines.push(line);
    });
    L.marker([data.origin.lat, data.origin.lon], {pane: 'routeFinder'}).bindTooltip('Origin').addTo(routeFinderLayer);
    L.marker([data.destination.lat, data.destination.lon], {pane: 'routeFinder'}).bindTooltip('Destination').addTo(routeFinderLayer);

    statusEl.textContent = data.unmatched_potholes
      ? `${data.unmatched_potholes} pothole report(s) too far from any road to route around.`
      : `${data.routes.length} route option(s) found — click one to highlight it.`;
    resultsEl.innerHTML = data.routes.map((r, i) => formatRoute(r, i, ROUTE_COLORS[i % ROUTE_COLORS.length])).join('');
    resultsEl.querySelectorAll('.route-option').forEach(el => {
      el.addEventListener('click', () => selectRoute(parseInt(el.dataset.index, 10)));
    });
    selectRoute(0);
  } catch (err) {
    statusEl.textContent = `Could not find a route: ${err.message}`;
  }
});

const car = L.marker([0, 0], {zIndexOffset: 1000, interactive: false, icon: L.divIcon({
  className: 'car-icon', iconSize: [42, 42], iconAnchor: [21, 21],
  html: '<div class="car-halo"><div class="car-arrow"><svg viewBox="0 0 24 24"><path d="m12 2 8 19-8-4-8 4Z"/></svg></div></div>',
})});
map.on('dragstart', () => setFollow(false));

function label(session) {
  if (session.display_name) return session.display_name;
  if (session.dataset === 'kaggle') return 'Kaggle · Larisa, Greece';
  const match = session.session_id.match(/gm_(\d+)_pass_(\d+)/);
  return `LiRA · M13 · Car ${match?.[1]} / Pass ${match?.[2]}`;
}

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
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
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
  const request = ++requestNumber;
  controller?.abort(); controller = new AbortController();
  setPlaying(false); loading = true;
  $('loading').hidden = false; $('load-error').hidden = true;
  $('play').disabled = true; $('seek').disabled = true;
  engine = null; clearMap(); visionImageKey = undefined; $('vision-panel').hidden = true;
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
    engine = new Replay(data); ordinal = ordinalSession(data.session); position = 0; loading = false;
    configureQuality();
    $('inference-status').hidden = data.session.inference_status !== 'blocked';
    $('inference-status').textContent = data.session.inference_status === 'blocked' ? 'Baseten inference pending: both deployments are inactive and the API rejects activation. GPS and sensor readings are available; road-quality and YOLO results have not been computed.' : '';
    $('about').hidden = data.session.inference_status === 'blocked';
    $('method-button').hidden = data.session.inference_status === 'blocked';
    $('vision-panel').hidden = !data.vision;
    $('annotated-video').hidden = !data.session.annotated_video_available;
    $('annotated-video').href = `/api/session/${encodeURIComponent(id)}/annotated.mp4`;
    $('next-pothole').disabled = !data.vision?.frames.some(f => f.detections.length);
    $('vision-summary').textContent = data.vision ? `${data.vision.provider === 'local_ultralytics' ? 'Local' : 'Baseten'} YOLO26 · ${data.vision.fps.toFixed(1)} fps · ${data.vision.frames.length.toLocaleString()} frames processed · ${data.vision.frames.filter(f => f.detections.length).length.toLocaleString()} frames with detections` : '';
    $('vision-alignment').textContent = data.vision ? (Number.isFinite(data.vision.video_offset_s) ? `Approximate video alignment: ${data.vision.video_offset_s >= 0 ? '+' : ''}${data.vision.video_offset_s.toFixed(3)} s from MP4 start metadata. Boxes show pothole confidence, not severity.` : 'Video alignment is unknown; detections are not assigned to the IMU timeline.') : '';
    $('session').value = id;
    $('profile').value = profile;
    $('profile-note').textContent = ordinal ? 'Saved calibration · Kalman Q/R 3.2 · 0.7 / 0.5 hysteresis' : profile === 'original' ? 'Consensus + hysteresis' : profile === 'threshold' ? 'Higher precision · lower recall' : 'Experimental · fewer alerts, more misses';
    $('model-line-note').textContent = profile === 'kalman' ? 'Solid: filtered · dashed: raw provisional' : 'Solid: final · dashed: provisional';
    document.querySelector('.download').textContent = 'Download outputs ↓';
    document.querySelector('.download').title = 'Download the full inference export, including predictions and input samples';
    $('session-meta').textContent = `${data.session.samples.toLocaleString()} samples · ${clock(data.session.duration_s)} drive · 100 Hz inputs`;
    $('duration').textContent = clock(data.duration_s);
    $('seek').max = String(data.duration_s);
    $('seek').disabled = false; $('play').disabled = false;
    $('loading').hidden = true;
    $('place').textContent = ordinal ? 'Recorded drive · Waterloo' : data.session.dataset === 'kaggle' ? 'Larisa, Greece' : 'M13 · Copenhagen, Denmark';
    const isLira = data.session.dataset === 'lira_cd';
    $('gyro-missing').hidden = !isLira;
    $('gyro-footer').textContent = isLira ? 'Missing input · not synthesized' : 'Recorded sensor axes';
    const providerLabel = data.session.inference_provider === 'local_pytorch' ? 'local PyTorch' : 'Baseten';
    $('task-note').textContent = ordinal ? `Original four-model ensemble · ${providerLabel}. Calibrated good / medium / bad estimates; these sessions have no ground-truth quality labels.${data.session.session_id === 'session5' ? ' Session 5 reuses the saved car bias; it was not in the original calibration set.' : ''}` : isLira
      ? 'LiRA provides measured roughness. Disturbance predictions have no reference labels here.'
      : 'Kaggle provides disturbance labels. Roughness estimates have no measured IRI reference here.';
    if (data.session.inference_status === 'blocked') $('task-note').textContent = 'Recorded GPS, accelerometer and speed only. No new cloud predictions have been produced.';
    document.querySelector('.page-footer > span').textContent = data.session.inference_status === 'blocked' ? 'Sensor replay · cloud inference pending' : 'Four-model ensemble · 10.24 s context · 320 ms finalization delay';
    if (data.gps.length) map.setView(data.gps[0].slice(1), 16, {animate: false});
    setFollow(true);
    seekTo(initialTime);
    history.replaceState(null, '', `?drive=${encodeURIComponent(id)}&profile=${encodeURIComponent(profile)}`);
    setPlaying(autoplay && !engine.ended);
  } catch (error) {
    if (error.name === 'AbortError' || request !== requestNumber) return;
    loading = false; $('loading').hidden = true; $('load-error').hidden = false;
    $('error-message').textContent = error.message;
    $('play').disabled = true; $('seek').disabled = true;
  }
}

function qualityColor(grade) { return ordinal ? ['#159c78', '#d2ae36', '#c95268'][grade] : COLORS[grade]; }
function configureQuality() {
  $('quality-unit').innerHTML = ordinal ? ' / 100<span class="unit-caption">roughness score</span>' : 'm/km<span class="unit-caption">estimated IRI</span>';
  $('quality-title').textContent = ordinal ? 'Calibrated road quality' : 'Overall road quality';
  document.querySelector('#model-chart').parentElement.querySelector('.axis-x').textContent = ordinal ? 'Quality score' : 'IRI';
  const scale = document.querySelector('.quality-scale');
  scale.querySelectorAll('span').forEach((el, i) => {el.hidden = ordinal && i === 3; el.style.background = qualityColor(i);});
  const legend = document.querySelector('.map-legend');
  legend.querySelectorAll('span').forEach(el => {if (el.textContent.trim() === 'Terrible') el.hidden = ordinal; if (el.textContent.trim() === 'Bad') el.querySelector('i').style.setProperty('--color',qualityColor(2));});
  document.querySelector('.probability-track > span').title = ordinal ? 'Onset threshold: 70%' : 'Onset threshold: 60%';
  document.querySelector('.probability-track > span').style.left = ordinal ? '70%' : '60%';
}
function renderVision() {
  const vision = engine?.data.vision;
  if (!vision) return;
  const frame = visionFrameAt(vision, engine.time);
  const key = frame ? `${engine.data.session.session_id}/${frame.frame_index}` : null;
  if (key === visionImageKey) return;
  visionImageKey = key;
  const image = $('vision-image'), overlay = $('vision-boxes');
  overlay.replaceChildren();
  $('vision-empty').hidden = !!frame; image.hidden = !frame;
  $('vision-time').textContent = frame ? `Video ${clock(frame.video_time_s,true)} · ${frame.detections.length} pothole detections` : 'No sampled video frame at this time';
  if (!frame) return;
  image.onload = () => { if (visionImageKey === key) image.style.opacity = '1'; };
  image.style.opacity = '0';
  image.src = `/api/session/${encodeURIComponent(engine.data.session.session_id)}/frames/${frame.frame_index}`;
  document.querySelector('.vision-stage').style.maxWidth = `${Math.min(960, frame.width / frame.height * 720)}px`;
  image.alt = `Road camera at ${clock(frame.video_time_s,true)}, ${frame.detections.length} pothole detections`;
  for (const d of frame.detections) {
    const [x1,y1,x2,y2] = d.box;
    const box = document.createElement('div'); box.className = 'vision-box';
    Object.assign(box.style,{left:`${100*x1/frame.width}%`,top:`${100*y1/frame.height}%`,width:`${100*(x2-x1)/frame.width}%`,height:`${100*(y2-y1)/frame.height}%`});
    const label = document.createElement('span'); label.textContent = `${d.label} ${(d.conf*100).toFixed(0)}%`; box.append(label); overlay.append(box);
  }
}

function paintQuality(row) {
  if (!row.is_final || !row.valid || !row.target_gps_valid) return;
  const index = row.target_gps_fix_index;
  const existing = qualityByFix.get(index);
  if (existing && existing.patch > row.target_patch) return;
  const fix = engine.data.gps[index], previous = engine.data.gps[index - 1];
  if (!fix) return;
  const options = {color: qualityColor(row.quality_grade), weight: 6, opacity: .88, pane: 'quality', renderer};
  const tooltip = `${qualityText(row, ordinal)}<br>Target ${clock(row.start_s, true)} · final at ${clock(row.available_s, true)}`;
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
    }).bindTooltip(`Provisional · ${qualityText(row, ordinal)}<br>No final disturbance decision yet`).addTo(pendingLayer);
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
  renderReadout(); drawCharts(); renderVision();
  if (engine.ended) setPlaying(false);
}

function renderReadout() {
  const final = engine.latestFinal, newest = engine.newest, gps = engine.gps;
  $('elapsed').textContent = clock(engine.time, true);
  $('seek').value = String(engine.time);
  const progress = engine.time / engine.data.duration_s * 100;
  $('seek').style.background = `linear-gradient(to right,#689977 ${progress}%,#e6ece3 ${progress}%)`;
  const label = engine.ended ? 'Drive complete · tail remains provisional' : playing ? 'Playing recorded test data' : 'Paused · explore the timeline';
  $('playback-label').textContent = label;
  $('map-status').innerHTML = `<i></i>${engine.ended ? 'DRIVE COMPLETE' : playing ? 'REPLAY IN PROGRESS' : 'REPLAY PAUSED'}`;
  $('coverage').textContent = `${Math.max(0, engine.sampleIndex + 1).toLocaleString()} / ${engine.data.session.samples.toLocaleString()} samples observed`;
  const rawOrigin = engine.data.session.timestamp_origin_unix_ns;
  const originMs = (rawOrigin != null && rawOrigin !== '') ? Number(rawOrigin) / 1e6 : null;
  if (originMs != null && !Number.isNaN(originMs) && Number.isFinite(originMs)) {
    const d = new Date(originMs + engine.time * 1000);
    if (!Number.isNaN(d.getTime())) {
      $('utc-clock').textContent = d.toISOString().replace('T', ' ').slice(0, 23) + ' UTC';
    } else {
      $('utc-clock').textContent = '—';
    }
  } else {
    $('utc-clock').textContent = '—';
  }
  const iri = qualityValue(final, ordinal);
  $('iri').textContent = iri == null ? '—' : iri.toFixed(ordinal ? 0 : 2);
  $('quality-badge').textContent = iri == null ? 'Waiting' : NAMES[final.quality_grade];
  $('quality-badge').style.color = iri == null ? '#74886b' : qualityColor(final.quality_grade);
  $('quality-badge').style.background = iri == null ? '#edf4ee' : qualityColor(final.quality_grade) + '14';
  $('quality-pointer').style.left = `${iri == null ? 0 : Math.min(99, iri / (ordinal ? 100 : 8) * 100)}%`;
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
  const chart = canvasSetup($('model-chart'), 0, ordinal ? 100 : 8, true);
  shadeEvents(chart);
  for (const provisional of [false, true]) {
    const use = rows.filter(row => row.is_final !== provisional).sort((a, b) => a.start_s - b.start_s);
    // Duplicate interval endpoints so patch estimates remain steps, not ramps.
    const iri = [], probability = [];
    for (const row of use) {
      for (const time of [row.start_s, Math.min(row.end_s, engine.time)]) {
        iri.push([time, qualityValue(row, ordinal)]);
        probability.push([time, row.valid ? row.probability : null]);
      }
    }
    drawLine(chart, iri, '#289b7b', 0, ordinal ? 100 : 8, provisional);
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
$('next-pothole').addEventListener('click', () => {
  const vision = engine?.data.vision;
  if (!vision || !Number.isFinite(vision.video_offset_s)) return;
  const next = vision.frames.find(f => f.detections.length && f.video_time_s + vision.video_offset_s > engine.time + .04);
  if (next) { setPlaying(false); seekTo(next.video_time_s + vision.video_offset_s + .001); $('vision-panel').scrollIntoView({behavior:'smooth',block:'center'}); }
});
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
  const response = await fetch('/api/catalog');
  if (!response.ok) throw new Error(`Catalog request failed: HTTP ${response.status}`);
  catalog = await response.json();
  $('session').replaceChildren();
  for (const dataset of [...new Set(catalog.sessions.map(s => s.dataset))]) {
    const group = document.createElement('optgroup');
    group.label = dataset === 'user_jsonl' ? 'Your recordings' : dataset === 'kaggle' ? 'Kaggle Road Quality' : 'LiRA-CD';
    for (const session of catalog.sessions.filter(s => s.dataset === dataset)) {
      const option = document.createElement('option'); option.value = session.session_id; option.textContent = label(session); group.append(option);
    }
    $('session').append(group);
  }
  $('session').disabled = false;
  $('profile').replaceChildren();
  for (const profile of catalog.profiles) {
    const option = document.createElement('option'); option.value = profile.id; option.textContent = profile.label;
    $('profile').append(option);
  }
  $('profile').disabled = false;
  fetchPotholes();
  const tiles = L.tileLayer(catalog.tile_url, {maxZoom: 19, attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors', keepBuffer: 2}).addTo(map);
  let failed = 0;
  tiles.on('tileerror', () => {failed++; if (failed >= 3) $('tile-warning').hidden = false;});
  tiles.on('tileload', () => {failed = 0; $('tile-warning').hidden = true;});
  const id = catalog.sessions.some(s => s.session_id === params.get('drive')) ? params.get('drive') : catalog.sessions[0].session_id;
  const profile = catalog.profiles.some(p => p.id === params.get('profile')) ? params.get('profile') : catalog.default_profile;
  await loadSession(id, Number(params.get('t')) || 0, params.get('autoplay') !== '0', profile);
} catch (error) {
  $('loading').hidden = true; $('load-error').hidden = false; $('error-message').textContent = error.message;
}
