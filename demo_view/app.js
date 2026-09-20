/**
 * RoadScope Pitch Demo View — Telemetry Playback & Road Quality Ledger
 * Built on `simulated_car_observations` table in Tiger PostgreSQL Cloud.
 * Implements 4-car scenario playback, color-coded road quality trails,
 * 15m spatial anomaly merging with confidence boosting,
 * and on-demand Dashcam Video & Accelerometer (IMU) Telemetry Inspection.
 */

import { resolveFleetVideo } from './fleet-video-sync.js';
import { createNavigationScenario } from './navigation-scenario.js';

let navigationScenario = null;
let scenarioLoadVersion = 0;
let scenarioLoading = false;
window.navigationSnapshot = () => navigationScenario?.snapshot() || null;

const API_BASE = window.location.port === '8888' ? 'http://localhost:8765' : window.location.origin;

// 4-Car Fleet Visual Configurations
const VEHICLE_CONFIGS = {
  '01': { icon: '🚗', color: '#df6e35', label: 'Car 01', title: 'Car 01 (Lead Patrol)' },
  '02': { icon: '🚌', color: '#2f5e38', label: 'Car 02', title: 'Car 02 (Transit Bus)' },
  '03': { icon: '🚐', color: '#1e88e5', label: 'Car 03', title: 'Car 03 (Courier Van)' },
  '04': { icon: '🚙', color: '#8e24aa', label: 'Car 04', title: 'Car 04 (City Service)' },
  'DEFAULT': { icon: '🚘', color: '#df6e35', label: 'Fleet Car', title: 'Municipal Fleet' }
};

function getVehicleConfig(carID) {
  if (!carID) return VEHICLE_CONFIGS['DEFAULT'];
  for (const key of ['01', '02', '03', '04']) {
    if (carID.endsWith(key) || carID.includes(`-${key}`)) {
      return VEHICLE_CONFIGS[key];
    }
  }
  return VEHICLE_CONFIGS['DEFAULT'];
}

// Global State
const state = {
  allObservations: [],
  vehicleRoutes: new Map(),        // carID -> [sorted observation waypoints]
  vehicleSegments: new Map(),      // carID -> [{ quality, strokeColor, weight, polyline, waypoints, startTs, endTs, isFull }]
  lastAnomalyPosPerVehicle: new Map(), // carID -> { lat, lon } (15m debouncing)
  potholes: [],
  potholeEvents: [],
  potholeEventIndex: 0,
  potholeReplayTime: -Infinity,
  syncManifest: null,
  selectedScenario: 'staggered',   // 'staggered' or 'cascade'
  selectedVehicle: 'ALL',
  isPlaying: false,
  playbackSpeed: 1.0,
  currentTimeMs: 0,
  startTimeMs: 0,
  endTimeMs: 0,
  ingestedAnomalyIds: new Set(),
  lastFrameTime: null,
  animationFrameId: null,
  dashcamFrame: 0,

  // On-demand Inspector State (±5s window)
  inspector: {
    isOpen: false,
    carID: 'sim-waterloo-2to5-staggered-01',
    centerTimeMs: 0,
    windowStartMs: 0,
    windowEndMs: 0,
    // relativeSec: -5.0,           // -5.0 to +5.0
    relativeSec: 0,
    isPlayingClip: false,
    isReady: false,
    clipStartVideoTime: 0,
    clipEndVideoTime: 10.0,
    loadToken: 0,
    animId: null,
    isIncident: false,
    severity: 'MEDIUM',
    imuSamples: [],
  }
};

// Map & Layer References
let map = null;
let trailLayer = null;
let carLayer = null;
let potholeLayer = null;
let vehicleMarkers = new Map();   // carID -> Leaflet Marker
let potholeMarkers = new Map();   // potholeId -> Leaflet Marker

// DOM Elements
const el = {
  btnPlayPause: document.getElementById('btn-play-pause'),
  playPauseText: document.getElementById('play-pause-text'),
  btnReset: document.getElementById('btn-reset'),
  btnNextAnomaly: document.getElementById('btn-next-anomaly'),
  btnSeed: document.getElementById('btn-seed'),
  btnOpenDrawer: document.getElementById('btn-open-drawer'),
  drawerCloseBtn: document.getElementById('drawer-close-btn'),
  drawerBackdrop: document.getElementById('drawer-backdrop'),
  dbDrawer: document.getElementById('db-drawer'),
  timelineSlider: document.getElementById('timeline-slider'),
  currentTimeLabel: document.getElementById('current-time-label'),
  totalTimeLabel: document.getElementById('total-time-label'),
  scenarioFilter: document.getElementById('scenario-filter'),
  vehicleFilter: document.getElementById('vehicle-filter'),
  toggleAutoIngest: document.getElementById('toggle-auto-ingest'),
  statVehicles: document.getElementById('stat-vehicles'),
  statWaypoints: document.getElementById('stat-waypoints'),
  statAnomalies: document.getElementById('stat-anomalies'),
  statPotholes: document.getElementById('stat-potholes'),
  dbStatusText: document.getElementById('db-status-text'),
  anomalyToast: document.getElementById('anomaly-toast'),
  toastCarId: document.getElementById('toast-car-id'),
  toastDesc: document.getElementById('toast-desc'),
  toastImuPill: document.getElementById('toast-imu-pill'),
  toastYoloPill: document.getElementById('toast-yolo-pill'),
  toastIngestStatus: document.getElementById('toast-ingest-status'),
  tabDetectionsBtn: document.getElementById('tab-detections-btn'),
  tabPotholesBtn: document.getElementById('tab-potholes-btn'),
  tabDetectionsContent: document.getElementById('tab-detections-content'),
  tabPotholesContent: document.getElementById('tab-potholes-content'),
  detectionsTableBody: document.getElementById('detections-table-body'),
  potholesTableBody: document.getElementById('potholes-table-body'),
  speedBtns: document.querySelectorAll('.speed-btn'),

  // Inspector Elements
  telemetryInspector: document.getElementById('telemetry-inspector'),
  inspectorBadge: document.getElementById('inspector-car-badge'),
  inspectorSubtitle: document.getElementById('inspector-subtitle'),
  btnCloseInspector: document.getElementById('btn-close-inspector'),
  inspectorVideo: document.getElementById('inspector-video'),
  inspectorCanvas: document.getElementById('inspector-canvas'),
  inspectorBuffering: document.getElementById('inspector-buffering'),
  inspectorBufferingText: document.getElementById('inspector-buffering-text'),
  inspectorHudLeft: document.getElementById('inspector-hud-left'),
  inspectorHudRight: document.getElementById('inspector-hud-right'),
  btnReplayClip: document.getElementById('btn-replay-clip'),
  btnLiveFeed: document.getElementById('btn-live-feed'),
  clipProgressFill: document.getElementById('clip-progress-fill'),
  chipAccelZ: document.getElementById('chip-accel-z'),
  chipAccelX: document.getElementById('chip-accel-x'),
  chipAccelY: document.getElementById('chip-accel-y'),
  accelWaveformCanvas: document.getElementById('accel-waveform-canvas'),
};

/**
 * Great-circle distance in meters between two lat/lon coordinates
 */
function getDistanceMeters(lat1, lon1, lat2, lon2) {
  const R = 6371000; // Earth radius in meters
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Normalizes severity levels
 */
function getSeverityLabel(sev) {
  if (sev === null || sev === undefined) return 'MEDIUM';
  if (typeof sev === 'number') {
    if (sev >= 8.5) return 'CRITICAL';
    if (sev >= 6.5) return 'HIGH';
    if (sev >= 4.0) return 'MEDIUM';
    return 'LOW';
  }
  const s = String(sev).toUpperCase().trim();
  if (s === 'CRITICAL' || s === 'HIGH' || s === 'MEDIUM' || s === 'LOW') return s;
  const num = parseFloat(s);
  if (!isNaN(num)) return getSeverityLabel(num);
  return 'MEDIUM';
}

/**
 * Merge any potholes within radiusMeters (15m) into a single canonical pothole
 * and boost confidence!
 */
function mergeDuplicatePotholes(potholesList, radiusMeters = 15.0) {
  const unique = [];
  for (const p of potholesList) {
    const lat = Number(p.latitude);
    const lon = Number(p.longitude);
    if (isNaN(lat) || isNaN(lon)) continue;

    let merged = false;
    for (const u of unique) {
      const dist = getDistanceMeters(lat, lon, Number(u.latitude), Number(u.longitude));
      if (dist <= radiusMeters) {
        // Boost confidence (+0.05 up to 0.99)
        const curConf = Number(u.confidence) || 0.85;
        const incConf = Number(p.confidence) || 0.85;
        u.confidence = Math.min(0.99, Math.round((Math.max(curConf, incConf) + 0.05) * 100) / 100);
        u.hit_count = (u.hit_count || 1) + 1;

        // Keep highest severity
        const sevOrder = { 'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 };
        const uSev = getSeverityLabel(u.severity);
        const pSev = getSeverityLabel(p.severity);
        if ((sevOrder[pSev] || 2) > (sevOrder[uSev] || 2)) {
          u.severity = pSev;
        }
        merged = true;
        break;
      }
    }
    if (!merged) {
      if (p.confidence == null) p.confidence = 0.85;
      if (p.hit_count == null) p.hit_count = 1;
      unique.push(p);
    }
  }
  return unique;
}

/**
 * Find closest active pothole within radiusMeters (15m)
 */
function findNearbyPothole(lat, lon, radiusMeters = 15.0) {
  let closest = null;
  let minDist = radiusMeters;
  for (const p of state.potholes) {
    const pLat = Number(p.latitude);
    const pLon = Number(p.longitude);
    if (isNaN(pLat) || isNaN(pLon)) continue;
    const d = getDistanceMeters(lat, lon, pLat, pLon);
    if (d <= minDist) {
      minDist = d;
      closest = p;
    }
  }
  return closest;
}

/**
 * Initialize Interactive Map with OpenStreetMap Cartography
 */
function initMap() {
  if (typeof L === 'undefined') {
    console.error('Leaflet failed to load');
    return;
  }

  map = L.map('map', {
    center: [43.4725, -80.5400],
    zoom: 14,
    zoomControl: false,
  });

  map.on('click', closeTelemetryInspector);
  L.control.zoom({ position: 'topleft' }).addTo(map);

  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19,
  }).addTo(map);

  trailLayer = L.featureGroup().addTo(map);
  potholeLayer = L.featureGroup().addTo(map);
  carLayer = L.featureGroup().addTo(map);

  setTimeout(() => {
    if (map) map.invalidateSize();
  }, 200);

  window.addEventListener('resize', () => {
    if (map) map.invalidateSize();
  });
}

/**
 * Fetch observations from simulated_car_observations and active potholes
 */
async function loadObservationsData() {
  const version = ++scenarioLoadVersion;
  const scenario = state.selectedScenario;
  scenarioLoading = true;
  state.isPlaying = false;
  cancelAnimationFrame(state.animationFrameId);
  if (el.playPauseText) el.playPauseText.textContent = '▶ Play Replay';
  closeTelemetryInspector();
  navigationScenario?.dispose();
  navigationScenario = null;
  for (const layer of [trailLayer, carLayer, potholeLayer]) layer?.clearLayers();
  potholeMarkers.clear();
  state.vehicleRoutes.clear();
  state.vehicleSegments.clear();
  state.potholes = [];
  state.allObservations = [];
  const isNavigation = ['warning', 'reroute'].includes(scenario);
  if (el.vehicleFilter) el.vehicleFilter.disabled = isNavigation;
  document.getElementById('nav-voice').onchange = () => navigationScenario?.stopAudio();
  try {
    if (isNavigation) {
      if (el.dbStatusText) el.dbStatusText.textContent = 'Loading navigation scenario…';
      const response = await fetch(`${API_BASE}/api/navigation-demo?mode=${scenario}`);
      if (!response.ok) throw new Error(`Scenario HTTP ${response.status}`);
      const data = await response.json();
      if (version !== scenarioLoadVersion) return;
      navigationScenario = createNavigationScenario(map, data, openNavigationHazard);
      state.startTimeMs = 0;
      state.currentTimeMs = 0;
      state.endTimeMs = navigationScenario.duration * 1000;
      el.timelineSlider.max = state.endTimeMs;
      el.timelineSlider.min = 0;
      el.timelineSlider.value = 0;
      el.totalTimeLabel.textContent = formatDuration(state.endTimeMs);
      el.currentTimeLabel.textContent = '00:00';
      el.statVehicles.textContent = `You + ${data.scouts.length} scouts`;
      el.statWaypoints.textContent = 'A → B';
      el.statAnomalies.textContent = '0';
      el.dbStatusText.textContent = 'Navigation demonstration · recorded fleet reports';
      populateDrawerTables();
      return;
    }
    if (el.dbStatusText) el.dbStatusText.textContent = 'Connecting...';

    // 1. Fetch Fleet Video Sync manifest
    try {
      const syncRes = await fetch(`${API_BASE}/api/fleet-sync`);
      if (syncRes.ok) {
        const manifest = await syncRes.json();
        if (version !== scenarioLoadVersion) return;
        state.syncManifest = manifest;
      }
    } catch (e) {
      console.warn('Could not load fleet-sync manifest:', e);
    }

    // 2. Fetch observations from simulated_car_observations table
    const obsUrl = `${API_BASE}/api/simulated-car-observations?scenario=${encodeURIComponent(state.selectedScenario)}&limit=50000`;
    const obsRes = await fetch(obsUrl);
    if (!obsRes.ok) throw new Error(`HTTP ${obsRes.status}`);
    const rawObs = await obsRes.json();
    if (version !== scenarioLoadVersion) return;
    state.allObservations = rawObs;

    scenarioLoading = false;
    // The persistent database contains discoveries from the entire drive.
    // Replay markers are reconstructed from timestamped observations instead.
    if (el.dbStatusText) el.dbStatusText.textContent = 'TigerDB (simulated_car_observations)';
    processObservations();
    renderPotholes();
    populateDrawerTables();
  } catch (err) {
    console.error('Failed to load observations data:', err);
    if (version === scenarioLoadVersion && el.dbStatusText) el.dbStatusText.textContent = 'Error loading scenario';
  } finally {
    if (version === scenarioLoadVersion) scenarioLoading = false;
  }
}

/**
 * Group observations by carID, sort chronologically, and create
 * persistent, high-performance polyline segments (zero lag during playback).
 */
function processObservations() {
  state.potholes = [];
  state.potholeEvents = [];
  state.potholeEventIndex = 0;
  state.potholeReplayTime = -Infinity;
  renderPotholes();
  state.vehicleRoutes.clear();
  state.vehicleSegments.clear();
  state.lastAnomalyPosPerVehicle.clear();
  if (trailLayer) trailLayer.clearLayers();

  if (!state.allObservations || state.allObservations.length === 0) {
    console.warn('No observations found in simulated_car_observations');
    return;
  }

  let minTs = Infinity;
  let maxTs = -Infinity;
  let anomalyCount = 0;
  const allCoords = [];

  state.allObservations.forEach((d) => {
    const carID = d.carID || d.car_id || 'sim-waterloo-01';
    if (!state.vehicleRoutes.has(carID)) {
      state.vehicleRoutes.set(carID, []);
    }
    state.vehicleRoutes.get(carID).push(d);

    if (d.timestamp < minTs) minTs = d.timestamp;
    if (d.timestamp > maxTs) maxTs = d.timestamp;
    if (d.imu_defect_detected && d.yolo_pothole_detected) {
      anomalyCount++;
    }

    if (d.latitude && d.longitude) {
      allCoords.push([d.latitude, d.longitude]);
    }
  });

  // A chronological event stream avoids skipping detections at high playback
  // speed or when scrubbing over a detection between animation frames.
  const lastDetection = new Map();
  state.potholeEvents = state.allObservations
    .filter(wp => wp.imu_defect_detected && wp.yolo_pothole_detected)
    .slice().sort((a, b) => a.timestamp - b.timestamp)
    .filter(wp => {
      const car = wp.carID || wp.car_id;
      const last = lastDetection.get(car);
      if (last && getDistanceMeters(wp.latitude, wp.longitude, last.latitude, last.longitude) < 15) return false;
      lastDetection.set(car, wp);
      return true;
    });

  // Sort each car's route chronologically
  state.vehicleRoutes.forEach((route) => {
    route.sort((a, b) => a.timestamp - b.timestamp);
  });

  state.startTimeMs = minTs;
  state.endTimeMs = maxTs;
  state.currentTimeMs = minTs;

  // Build contiguous quality segments per car
  // This turns 7,000 separate polylines into ~30 contiguous segments per car!
  state.vehicleRoutes.forEach((waypoints, carID) => {
    const segments = [];
    let curSeg = null;

    for (let i = 0; i < waypoints.length; i++) {
      const wp = waypoints[i];
      let q = (wp.road_quality || 'good').toLowerCase();
      if (wp.imu_defect_detected && wp.yolo_pothole_detected) {
        q = 'bad';
      } else if (wp.imu_defect_detected || wp.yolo_pothole_detected) {
        if (q === 'good') q = 'medium';
      }

      if (!curSeg || curSeg.quality !== q) {
        if (curSeg && curSeg.waypoints.length > 0) {
          curSeg.waypoints.push(wp); // connect without gaps
          curSeg.endTs = wp.timestamp;
        }
        const strokeColor = (q === 'bad') ? '#c62828' : (q === 'medium' ? '#e65100' : '#2e7d32');
        const weight = (q === 'bad') ? 7 : (q === 'medium' ? 6 : 5);

        const polyline = L.polyline([], {
          color: strokeColor,
          weight: weight,
          opacity: 0.9,
          lineCap: 'round',
          lineJoin: 'round',
        });

        // Clicking on a road polyline segment inspects that point!
        polyline.on('click', (e) => {
          L.DomEvent.stopPropagation(e);
          openTelemetryInspectorForPoint(carID, wp.timestamp, wp.latitude, wp.longitude, q === 'bad', q.toUpperCase());
        });

        // Attach only after this road is reached; future roads need no SVG elements.

        curSeg = {
          quality: q,
          strokeColor: strokeColor,
          weight: weight,
          polyline: polyline,
          waypoints: [wp],
          startTs: wp.timestamp,
          endTs: wp.timestamp,
          isFull: false,
        };
        segments.push(curSeg);
      } else {
        curSeg.waypoints.push(wp);
        curSeg.endTs = wp.timestamp;
      }
    }
    state.vehicleSegments.set(carID, segments);
  });

  // Update Stats Counters
  if (el.statVehicles) el.statVehicles.textContent = `${state.vehicleRoutes.size} Cars`;
  if (el.statWaypoints) el.statWaypoints.textContent = state.allObservations.length.toLocaleString();
  if (el.statAnomalies) el.statAnomalies.textContent = anomalyCount.toLocaleString();
  if (el.statPotholes) el.statPotholes.textContent = state.potholes.length;

  // Update Timeline Scrubber
  const totalDurationMs = Math.max(1000, state.endTimeMs - state.startTimeMs);
  if (el.timelineSlider) {
    el.timelineSlider.min = 0;
    el.timelineSlider.max = totalDurationMs;
    el.timelineSlider.value = 0;
  }
  if (el.totalTimeLabel) el.totalTimeLabel.textContent = formatDuration(totalDurationMs);
  if (el.currentTimeLabel) el.currentTimeLabel.textContent = '00:00';

  // Frame map to view all car routes
  if (map && allCoords.length > 0) {
    map.fitBounds(allCoords, { padding: [50, 50], maxZoom: 16 });
  }

  initVehicleMarkers();
  updateMapToCurrentTime();
}

/**
 * Initialize Leaflet markers for the 4 cars
 */
function initVehicleMarkers() {
  if (!carLayer) return;
  carLayer.clearLayers();
  vehicleMarkers.clear();

  state.vehicleRoutes.forEach((waypoints, carID) => {
    if (waypoints.length === 0) return;
    const firstWp = waypoints[0];
    const cfg = getVehicleConfig(carID);

    const iconHtml = `
      <div class="car-marker-container">
        <div class="car-icon-token" style="background: ${cfg.color}">
          <span>${cfg.icon}</span>
        </div>
        <div class="car-label-pill">${cfg.label}</div>
      </div>
    `;

    const marker = L.marker([firstWp.latitude, firstWp.longitude], {
      icon: L.divIcon({
        className: 'custom-car-div-icon',
        html: iconHtml,
        iconSize: [40, 60],
        iconAnchor: [20, 20],
      }),
      zIndexOffset: 1000,
    });

    // Clicking on car marker opens the Video & Accelerometer Telemetry Inspector!
    marker.on('click', (e) => {
      L.DomEvent.stopPropagation(e);
      openTelemetryInspectorForCar(carID);
    });

    marker.bindPopup(`
      <div style="font-family: var(--font-serif); font-size: 12px; color: #3d2816; min-width: 170px;">
        <strong style="color: ${cfg.color}">${cfg.icon} ${cfg.title}</strong><br>
        <strong>Car ID:</strong> <code>${carID}</code><br>
        <strong>GPS:</strong> ${firstWp.latitude.toFixed(5)}, ${firstWp.longitude.toFixed(5)}<br>
        <strong>Quality:</strong> ${firstWp.road_quality || 'good'}<br>
        <button onclick="window.inspectVehicle('${carID}')" style="margin-top: 6px; width: 100%; background: ${cfg.color}; color: #fff; border: none; border-radius: 4px; padding: 4px 8px; font-weight: bold; cursor: pointer;">
          📹 Inspect Dashcam & Telemetry
        </button>
      </div>
    `);

    carLayer.addLayer(marker);
    vehicleMarkers.set(carID, marker);
  });
}

// Window hook for popup button
window.inspectVehicle = (carID) => {
  openTelemetryInspectorForCar(carID);
};

/**
 * Update single pothole marker with live confidence & pulse effect
 */
function updatePotholeMarker(p, pulse = false) {
  const pId = String(p.id || `${Number(p.latitude).toFixed(4)}_${Number(p.longitude).toFixed(4)}`);
  const lat = Number(p.latitude);
  const lon = Number(p.longitude);
  if (isNaN(lat) || isNaN(lon)) return;

  const sev = getSeverityLabel(p.severity);
  const conf = p.confidence ? Math.round(Number(p.confidence) * 100) : 85;
  const hits = p.hit_count || 1;

  let markerColor = '#d88b32';
  if (sev === 'CRITICAL') markerColor = '#b83824';
  else if (sev === 'HIGH') markerColor = '#df6e35';
  else if (sev === 'LOW') markerColor = '#4c7847';

  const pulseClass = pulse ? 'merged-pulse' : '';
  const potholeHtml = `
    <div class="pothole-stamp-marker ${pulseClass}" style="background: ${markerColor};" title="Registered Pothole: ${sev} (${conf}% conf, ${hits} detections)">
      ⚠️
    </div>
  `;

  const popupContent = `
    <div style="font-family: var(--font-serif); font-size: 12px; color: #3d2816; min-width: 175px;">
      <strong style="color: ${markerColor}">⚠️ Registered Pothole (${sev})</strong><br>
      <strong>Confidence:</strong> <span style="font-size: 13px; font-weight: 800; color: ${conf >= 90 ? '#b83824' : '#3b6138'};">${conf}%</span><br>
      <strong>Sensor Detections:</strong> ${hits} ${hits > 1 ? '<span style="color: #df6e35; font-weight: bold;">(15m Merged)</span>' : ''}<br>
      <strong>GPS:</strong> ${lat.toFixed(5)}, ${lon.toFixed(5)}<br>
      <button onclick="window.inspectPothole(${lat}, ${lon})" style="margin-top: 6px; width: 100%; background: #df6e35; color: #fff; border: none; border-radius: 4px; padding: 4px 8px; font-weight: bold; cursor: pointer;">
        ⚡ Inspect ±5s Impact Window
      </button>
    </div>
  `;

  if (potholeMarkers.has(pId)) {
    const m = potholeMarkers.get(pId);
    m.setLatLng([lat, lon]);
    m.setIcon(L.divIcon({
      className: 'custom-pothole-div-icon',
      html: potholeHtml,
      iconSize: [26, 26],
      iconAnchor: [13, 13],
    }));
    m.setPopupContent(popupContent);
    if (pulse) {
      const domEl = m.getElement();
      if (domEl) {
        const inner = domEl.querySelector('.pothole-stamp-marker');
        if (inner) {
          inner.classList.remove('merged-pulse');
          void inner.offsetWidth;
          inner.classList.add('merged-pulse');
        }
      }
    }
  } else {
    const pMarker = L.marker([lat, lon], {
      icon: L.divIcon({
        className: 'custom-pothole-div-icon',
        html: potholeHtml,
        iconSize: [26, 26],
        iconAnchor: [13, 13],
      }),
    });

    pMarker.on('click', (e) => {
      L.DomEvent.stopPropagation(e);
      openTelemetryInspectorForPoint(null, null, lat, lon, true, sev);
    });

    pMarker.bindPopup(popupContent);
    potholeLayer.addLayer(pMarker);
    potholeMarkers.set(pId, pMarker);
  }
}

window.inspectPothole = (lat, lon) => {
  openTelemetryInspectorForPoint(null, null, lat, lon, true, 'CRITICAL');
};

/**
 * Render registered potholes layer with marker reuse
 */
function renderPotholes() {
  if (!potholeLayer) return;

  const activeIds = new Set(state.potholes.map(p => String(p.id || `${Number(p.latitude).toFixed(4)}_${Number(p.longitude).toFixed(4)}`)));
  potholeMarkers.forEach((marker, id) => {
    if (!activeIds.has(id)) {
      potholeLayer.removeLayer(marker);
      potholeMarkers.delete(id);
    }
  });

  state.potholes.forEach((p) => {
    updatePotholeMarker(p, false);
  });

  if (el.statPotholes) el.statPotholes.textContent = state.potholes.length;
}

/**
 * Core Playback Render: updates vehicle markers & persistent trails with zero lag!
 */
function updateMapToCurrentTime(announce = false) {
  if (scenarioLoading) return;
  if (navigationScenario) {
    navigationScenario.render(state.currentTimeMs / 1000, announce);
    el.timelineSlider.value = state.currentTimeMs;
    const label = formatDuration(state.currentTimeMs);
    if (el.currentTimeLabel.textContent !== label) el.currentTimeLabel.textContent = label;
    return;
  }
  syncReplayPotholes();
  const cars = Array.from(state.vehicleRoutes.keys()).sort();

  cars.forEach((carID) => {
    const isVisible = (state.selectedVehicle === 'ALL' || carID.includes(state.selectedVehicle));
    const marker = vehicleMarkers.get(carID);
    const waypoints = state.vehicleRoutes.get(carID) || [];
    const segments = state.vehicleSegments.get(carID) || [];
    if (waypoints.length === 0) return;

    const hasLaunched = (waypoints[0].timestamp <= state.currentTimeMs);

    if (!isVisible || !hasLaunched) {
      if (marker && carLayer.hasLayer(marker)) {
        carLayer.removeLayer(marker);
      }
      segments.renderedIndex = undefined;
      segments.forEach((seg) => {
        if (seg.isFull || seg.polyline.getLatLngs().length > 0) {
          seg.polyline.setLatLngs([]);
          trailLayer.removeLayer(seg.polyline);
          seg.isFull = false;
          seg.lastWpIdx = -1;
        }
      });
      return;
    } else {
      if (marker && !carLayer.hasLayer(marker)) {
        carLayer.addLayer(marker);
      }
    }

    // Binary search to find current waypoint index
    let passedIndex = -1;
    let low = 0, high = waypoints.length - 1;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (waypoints[mid].timestamp <= state.currentTimeMs) {
        passedIndex = mid;
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }

    let currentPos = null;
    if (passedIndex >= 0) {
      const currWp = waypoints[passedIndex];
      if (passedIndex < waypoints.length - 1) {
        const nextWp = waypoints[passedIndex + 1];
        const span = nextWp.timestamp - currWp.timestamp;
        const frac = span > 0 ? (state.currentTimeMs - currWp.timestamp) / span : 0;
        const clamped = Math.max(0, Math.min(1, frac));
        currentPos = [
          currWp.latitude + (nextWp.latitude - currWp.latitude) * clamped,
          currWp.longitude + (nextWp.longitude - currWp.longitude) * clamped,
        ];
      } else {
        currentPos = [currWp.latitude, currWp.longitude];
      }

    } else {
      currentPos = [waypoints[0].latitude, waypoints[0].longitude];
    }

    // Update car marker position
    if (marker && currentPos) {
      marker.setLatLng(currentPos);
    }

    // GPS waypoints change much less often than animation frames. Keep marker
    // interpolation smooth, but only revisit road geometry after a new waypoint.
    if (segments.renderedIndex === passedIndex && segments.renderedVisible === isVisible) return;
    segments.renderedIndex = passedIndex;
    segments.renderedVisible = isVisible;
    segments.forEach((seg) => {
      if (seg.startTs > state.currentTimeMs) {
        // Future segment
        if (seg.isFull || seg.polyline.getLatLngs().length > 0) {
          seg.polyline.setLatLngs([]);
          trailLayer.removeLayer(seg.polyline);
          seg.isFull = false;
          seg.lastWpIdx = -1;
        }
      } else if (seg.endTs <= state.currentTimeMs) {
        // Fully traversed segment: set once and keep
        if (!seg.isFull) {
          const pts = [];
          const stride = seg.waypoints.length > 40 ? 2 : 1;
          for (let k = 0; k < seg.waypoints.length; k += stride) {
            pts.push([seg.waypoints[k].latitude, seg.waypoints[k].longitude]);
          }
          const lastWp = seg.waypoints[seg.waypoints.length - 1];
          pts.push([lastWp.latitude, lastWp.longitude]);
          seg.polyline.setLatLngs(pts);
          if (!trailLayer.hasLayer(seg.polyline)) trailLayer.addLayer(seg.polyline);
          seg.isFull = true;
        }
      } else {
        // Currently active segment: only update geometry when car passes new waypoint (stops lag!)
        if (seg.lastWpIdx !== passedIndex) {
          seg.lastWpIdx = passedIndex;
          seg.isFull = false;
          const pts = [];
          const stride = seg.waypoints.length > 40 ? 2 : 1;
          for (let k = 0; k < seg.waypoints.length; k += stride) {
            const wp = seg.waypoints[k];
            if (wp.timestamp <= state.currentTimeMs) {
              pts.push([wp.latitude, wp.longitude]);
            } else {
              break;
            }
          }
          if (currentPos) pts.push(currentPos);
          seg.polyline.setLatLngs(pts);
          if (!trailLayer.hasLayer(seg.polyline)) trailLayer.addLayer(seg.polyline);
        }
      }
    });
  });

  // Update Scrubber UI
  const elapsedMs = state.currentTimeMs - state.startTimeMs;
  if (el.timelineSlider) el.timelineSlider.value = Math.max(0, elapsedMs);
  const timeLabel = formatDuration(elapsedMs);
  if (el.currentTimeLabel && el.currentTimeLabel.textContent !== timeLabel) el.currentTimeLabel.textContent = timeLabel;

}

/**
 * Invalidate cached full polylines on scrub / jump
 */
function invalidateSegmentCaches() {
  state.vehicleSegments.forEach((segments) => {
    segments.renderedIndex = undefined;
    segments.forEach((seg) => {
      seg.isFull = false;
      seg.lastWpIdx = -1;
    });
  });
}

/**
 * Replay discoveries using observation time, never the database's final state.
 * Rewind rebuilds only past evidence; forward playback consumes each event once.
 * Viewing a recording must not add duplicate detections to the live database.
 */
function syncReplayPotholes() {
  const now = state.currentTimeMs;
  let changed = false;
  if (now < state.potholeReplayTime) {
    state.potholes = [];
    state.potholeEventIndex = 0;
    changed = true;
  }
  while (state.potholeEventIndex < state.potholeEvents.length &&
         state.potholeEvents[state.potholeEventIndex].timestamp <= now) {
    const wp = state.potholeEvents[state.potholeEventIndex++];
    const nearby = findNearbyPothole(wp.latitude, wp.longitude, 15);
    if (nearby) {
      nearby.hit_count++;
      nearby.confidence = Math.min(0.99, Math.round((nearby.confidence + 0.05) * 100) / 100);
    } else {
      state.potholes.push({
        id: `replay-${state.potholeEventIndex}`,
        latitude: wp.latitude, longitude: wp.longitude,
        severity: 'CRITICAL', confidence: 0.85, hit_count: 1,
        timestamp: new Date(wp.timestamp).toISOString(),
        detected_by_vision: true, detected_by_imu: true,
      });
    }
    changed = true;
  }
  state.potholeReplayTime = now;
  if (changed) {
    renderPotholes();
    populateDrawerTables();
  }
}

/* ==========================================================================
   Telemetry & Dashcam Inspector (±5.0s Video & 3-Axis Mock Telemetry)
   ========================================================================== */

/**
 * Stop any running inspector animation loop and pause video
 */
let inspectorFetch = null;
let inspectorMedia = null;
const imuWindowCache = new Map();
async function fetchInspectorWindow(url) {
  inspectorFetch?.abort();
  inspectorFetch = new AbortController();
  if (imuWindowCache.has(url)) return imuWindowCache.get(url);
  const response = await fetch(url, {signal: inspectorFetch.signal});
  if (!response.ok) throw new Error(`Telemetry unavailable (${response.status})`);
  const data = await response.json();
  imuWindowCache.set(url, data);
  if (imuWindowCache.size > 48) imuWindowCache.delete(imuWindowCache.keys().next().value);
  return data;
}

function stopInspectorLoop() {
  inspectorFetch?.abort();
  inspectorMedia?.abort();
  if (state.inspector.animId) {
    cancelAnimationFrame(state.inspector.animId);
    state.inspector.animId = null;
  }
  if (el.inspectorVideo && !el.inspectorVideo.paused) {
    el.inspectorVideo.pause();
  }
}

/**
 * Close the inspector drawer/panel
 */
async function openNavigationHazard(hazard) {
  closeTelemetryInspector();
  const token = ++state.inspector.loadToken;
  state.inspector.isOpen = true;
  el.telemetryInspector.style.display = 'flex';
  el.inspectorSubtitle.textContent = `Loading ${hazard.source_session} at ${hazard.source_time_s.toFixed(2)}s…`;
  el.inspectorVideo.style.display = 'none';
  state.inspector.imuSamples = [];
  if (el.btnLiveFeed) el.btnLiveFeed.hidden = true;
  try {
    const source = await fetchInspectorWindow(`${API_BASE}/api/session/${encodeURIComponent(hazard.source_session)}/imu-window?time_s=${hazard.source_time_s}`);
    if (!state.inspector.isOpen || token !== state.inspector.loadToken) return;
    openTelemetryInspectorForPoint(hazard.scout_id, hazard.source_time_s * 1000,
      ...hazard.coords, true, 'CRITICAL', source);
    el.inspectorBadge.textContent = `${hazard.scout_id} · ${source.session}`;
    el.inspectorSubtitle.textContent = `Recorded ${source.session} · ${source.center_s.toFixed(2)}s · ±5s`;
  } catch (error) {
    if (token === state.inspector.loadToken && state.inspector.isOpen) el.inspectorSubtitle.textContent = error.message;
  }
}

function closeTelemetryInspector() {
  state.inspector.source = null;
  if (el.btnLiveFeed) el.btnLiveFeed.hidden = false;
  state.inspector.isOpen = false;
  state.inspector.isIncident = false;
  state.inspector.loadToken++;
  stopInspectorLoop();
  if (el.telemetryInspector) el.telemetryInspector.style.display = 'none';
  if (el.inspectorBuffering) el.inspectorBuffering.style.display = 'none';
}

/**
 * Fetch real high-frequency telemetry from PostgreSQL table: simulated_car_imu_samples
 */
async function loadRealImuSamples(carID, centerTimeMs) {
  const token = state.inspector.loadToken;
  // A moving replay can advance faster than a request completes. Do not abort
  // this selection's in-flight refresh, or slow connections never get samples.
  if (state.inspector.imuPendingToken === token) return;
  state.inspector.imuPendingToken = token;
  try {
    const data = await fetchInspectorWindow(`${API_BASE}/api/imu-samples?car_id=${encodeURIComponent(carID)}&timestamp=${centerTimeMs}&window_seconds=5.0`);
    if (token !== state.inspector.loadToken || !state.inspector.isOpen) return;
    state.inspector.imuSamples = data.samples || [];
    state.inspector.sampleCenterMs = centerTimeMs;
    renderRealAccelerometer(state.inspector.isIncident ? state.inspector.relativeSec :
      (state.currentTimeMs-centerTimeMs)/1000, state.inspector.severity);
    return;
  } catch (e) {
    if (e.name === 'AbortError') return;
    console.warn('Failed to load IMU samples:', e);
  } finally {
    if (state.inspector.imuPendingToken === token) state.inspector.imuPendingToken = null;
  }
  if (token === state.inspector.loadToken) state.inspector.imuSamples = [];
}

/**
 * Prepares the ±5.0s incident clip:
 * WAITS for the video to be ready (metadata loaded + seeked to start frame)
 * BEFORE starting playback or ticking the clock!
 */
function prepareAndPlayIncidentClip() {
  stopInspectorLoop();

  state.inspector.loadToken++;
  const token = state.inspector.loadToken;
  inspectorMedia = new AbortController();
  const mediaSignal = inspectorMedia.signal;

  const vid = el.inspectorVideo;
  if (!vid) return;

  // Show buffering overlay
  if (el.inspectorBuffering) {
    if (el.inspectorBufferingText) el.inspectorBufferingText.textContent = 'Aligning ±5.0s Video Clip...';
    el.inspectorBuffering.style.display = 'flex';
  }

  // Reset relativeSec to -5.0s and update UI immediately
  state.inspector.relativeSec = -5.0;
  if (el.clipProgressFill) el.clipProgressFill.style.width = '0%';
  if (el.inspectorHudRight) el.inspectorHudRight.textContent = 'BUFFERING ±5.0s...';
  renderRealAccelerometer(-5.0, state.inspector.severity);

  // Fetch real database IMU records for this vehicle in the ±5.0s window
  if (state.inspector.source) state.inspector.imuSamples = state.inspector.source.samples;
  else loadRealImuSamples(state.inspector.carID, state.inspector.centerTimeMs);

  // Compute start video position from sync manifest (-5.0s from impact center)
  const sync = state.syncManifest;
  const source = state.inspector.source;
  const startTarget = source ? {
    session: source.session,
    url: `/demo_view/videos/${encodeURIComponent(source.session)}.mp4`,
    currentTime: source.center_s - 5 - source.video_offset_s,
  } : resolveFleetVideo(sync, state.selectedScenario, state.inspector.carID, state.inspector.windowStartMs);
  const expectedUrl = startTarget.url || `/demo_view/videos/session2.mp4`;
  const targetStartTime = Math.max(0, startTarget.currentTime);

  state.inspector.clipStartVideoTime = targetStartTime;
  state.inspector.clipEndVideoTime = targetStartTime + 10.0;

  const cfg = getVehicleConfig(state.inspector.carID);
  if (el.inspectorHudLeft) {
    el.inspectorHudLeft.textContent = `REC // ${cfg.label} // ${(startTarget.session || 'SESSION 2').toUpperCase()}`;
  }

  // Callback once the video has reached the target frame and is ready
  let started = false;
  const onVideoReady = () => {
    if (started || mediaSignal.aborted) return;
    if (token !== state.inspector.loadToken || !state.inspector.isOpen || !state.inspector.isIncident) return;

    if (vid.readyState < 2) {
      vid.addEventListener('canplay', onVideoReady, {once:true, signal:mediaSignal});
      return;
    }
    started = true;
    if (el.inspectorBuffering) el.inspectorBuffering.style.display = 'none';
    vid.style.display = 'block';
    if (el.inspectorCanvas) el.inspectorCanvas.style.display = 'none';

    // Apply speed setting to native HTML5 video element
    vid.playbackRate = state.playbackSpeed;

    // Start video playback
    const playPromise = vid.play();
    if (playPromise) {
      playPromise.catch((e) => console.warn('Inspector clip play caught:', e));
    }

    // Start animation loop in lockstep with the video
    startIncidentLoop(token);
  };

  const seekToStart = () => {
    // If already at or very close to start time and ready, proceed
    if (Math.abs(vid.currentTime - targetStartTime) < 0.08 && vid.readyState >= 2) {
      onVideoReady();
    } else {
      let fired = false;
      const onSeeked = () => {
        if (fired) return;
        fired = true;
        vid.removeEventListener('seeked', onSeeked);
        onVideoReady();
      };
      vid.addEventListener('seeked', onSeeked, { once: true, signal: mediaSignal });
      vid.currentTime = targetStartTime;

    }
  };

  // currentSrc may be empty until resource selection finishes. Compare the
  // assigned URL so a slow load is not restarted on every animation frame.
  const sourceChanged = vid.src !== new URL(expectedUrl, document.baseURI).href;

  if (sourceChanged) {
    // Switch video file source
    vid.pause();
    vid.addEventListener('loadedmetadata', () => {
      if (token !== state.inspector.loadToken) return;
      seekToStart();
    }, { once: true, signal: mediaSignal });
    vid.src = expectedUrl;
    vid.load();
  } else {
    // Already correct video source
    if (vid.readyState >= 1) {
      seekToStart();
    } else {
      vid.addEventListener('loadedmetadata', () => {
        if (token !== state.inspector.loadToken) return;
        seekToStart();
      }, { once: true, signal: mediaSignal });
    }
  }
}

/**
 * Loop the ±5.0s incident clip:
 * Derived directly from vid.currentTime so speed (0.5x, 1x, 2x, 5x)
 * runs natively in hardware without drift, lag, or periodic seeking!
 */
function startIncidentLoop(token) {
  const vid = el.inspectorVideo;

  function tick() {
    if (token !== state.inspector.loadToken || !state.inspector.isOpen || !state.inspector.isIncident) {
      return;
    }

    // Keep playbackRate strictly matched
    if (vid && vid.playbackRate !== state.playbackSpeed) {
      vid.playbackRate = state.playbackSpeed;
    }

    // Calculate relativeSec directly from the video element's actual position
    if (vid && !vid.paused && vid.currentTime >= state.inspector.clipStartVideoTime) {
      const elapsed = vid.currentTime - state.inspector.clipStartVideoTime;
      state.inspector.relativeSec = -5.0 + elapsed;
    }

    const rel = state.inspector.relativeSec;

    // Update HUD text
    if (el.inspectorHudRight) {
      const relSign = rel >= 0 ? `+${rel.toFixed(1)}` : rel.toFixed(1);
      const isImpact = Math.abs(rel) < 0.4;
      el.inspectorHudRight.textContent = `OFFSET: ${relSign}s ${isImpact ? '💥 [IMPACT]' : ''}`;
    }

    // Update Progress Fill across 10-second window (-5.0s to +5.0s)
    if (el.clipProgressFill) {
      const pct = Math.max(0, Math.min(100, ((rel + 5.0) / 10.0) * 100));
      el.clipProgressFill.style.width = `${pct}%`;
    }

    // Render Accelerometer Telemetry across 10-second window
    renderMockAccelerometer(rel, state.inspector.severity);

    // When clip reaches +5.0s, loop cleanly back to -5.0s:
    // Wait for video to seek back before replaying!
    if (rel >= 5.0 || (vid && vid.currentTime >= state.inspector.clipEndVideoTime)) {
      if (vid) vid.pause();
      prepareAndPlayIncidentClip();
      return;
    }

    state.inspector.animId = requestAnimationFrame(tick);
  }

  state.inspector.animId = requestAnimationFrame(tick);
}

/**
 * Open Inspector focused on a specific road point or pothole impact (±5.0s window)
 */
function openTelemetryInspectorForPoint(carID, timestamp, lat, lon, isIncident = true, sev = 'CRITICAL', source = null) {
  state.inspector.source = source;
  state.inspector.imuSamples = source?.samples || [];
  if (el.btnLiveFeed) el.btnLiveFeed.hidden = Boolean(source);
  let chosenCar = carID;
  let chosenTs = timestamp;

  if (!chosenCar || !chosenTs) {
    let closestDist = Infinity;
    state.vehicleRoutes.forEach((waypoints, cId) => {
      for (const wp of waypoints) {
        const d = getDistanceMeters(lat, lon, wp.latitude, wp.longitude);
        if (d < closestDist) {
          closestDist = d;
          chosenCar = cId;
          chosenTs = wp.timestamp;
        }
      }
    });
  }

  if (!chosenCar) {
    chosenCar = Array.from(state.vehicleRoutes.keys())[0] || 'sim-waterloo-2to5-staggered-01';
    chosenTs = state.currentTimeMs;
  }

  const cfg = getVehicleConfig(chosenCar);
  state.inspector.isOpen = true;
  state.inspector.carID = chosenCar;
  state.inspector.isIncident = isIncident;
  state.inspector.centerTimeMs = chosenTs || state.currentTimeMs;
  state.inspector.windowStartMs = state.inspector.centerTimeMs - 5000;
  state.inspector.windowEndMs = state.inspector.centerTimeMs + 5000;
  state.inspector.relativeSec = -5.0;
  state.inspector.severity = sev;

  if (el.telemetryInspector) el.telemetryInspector.style.display = 'flex';
  if (el.inspectorBadge) {
    el.inspectorBadge.textContent = `${cfg.icon} ${cfg.label}`;
    el.inspectorBadge.style.background = cfg.color;
  }
  if (el.inspectorSubtitle) {
    el.inspectorSubtitle.textContent = isIncident
      ? `±5.0s Impact Incident Window (${sev})`
      : `±5.0s Waypoint Telemetry Window`;
  }

  prepareAndPlayIncidentClip();
}

/**
 * Open Inspector focused on a specific car (tracks car in real time)
 */
function openTelemetryInspectorForCar(carID) {
  state.inspector.imuSamples = [];
  state.inspector.sampleCenterMs = null;
  state.inspector.source = null;
  if (el.btnLiveFeed) el.btnLiveFeed.hidden = false;
  const cfg = getVehicleConfig(carID);
  state.inspector.isOpen = true;
  state.inspector.carID = carID;
  state.inspector.isIncident = false;
  state.inspector.centerTimeMs = state.currentTimeMs;
  state.inspector.relativeSec = 0.0;
  state.inspector.severity = 'MEDIUM';

  if (el.telemetryInspector) el.telemetryInspector.style.display = 'flex';
  if (el.inspectorBadge) {
    el.inspectorBadge.textContent = `${cfg.icon} ${cfg.label}`;
    el.inspectorBadge.style.background = cfg.color;
  }
  if (el.inspectorSubtitle) {
    el.inspectorSubtitle.textContent = `Live Dashcam & 3-Axis Telemetry`;
  }

  startLiveDrivingInspector();
}

/**
 * Live vehicle driving mode: follows main simulation clock smoothly
 */
function startLiveDrivingInspector() {
  stopInspectorLoop();

  state.inspector.loadToken++;
  const token = state.inspector.loadToken;

  const vid = el.inspectorVideo;
  if (!vid) return;

  if (el.inspectorBuffering) el.inspectorBuffering.style.display = 'none';
  vid.style.display = 'block';
  if (el.inspectorCanvas) el.inspectorCanvas.style.display = 'none';

  inspectorMedia = new AbortController();
  // Resolve against the current car clock when loading finishes, not the time
  // at which the request began. Selection changes cancel these listeners.
  for (const event of ['loadedmetadata', 'canplay', 'seeked']) {
    vid.addEventListener(event, () => {
      if (token === state.inspector.loadToken && state.inspector.isOpen && !state.inspector.isIncident) {
        syncLiveInspectorVideo();
      }
    }, {signal: inspectorMedia.signal});
  }
  syncLiveInspectorVideo(true);
  loadRealImuSamples(state.inspector.carID, state.currentTimeMs);

  let lastFetchTime = state.currentTimeMs;
  let lastTick = -Infinity;
  function liveTick(timestamp) {
    if (token !== state.inspector.loadToken || !state.inspector.isOpen || state.inspector.isIncident) {
      return;
    }

    // Native video playback supplies the frames. Alignment only needs 10 Hz,
    // including while paused so scrubs and pause changes are reflected promptly.
    if (timestamp - lastTick >= 100) {
      lastTick = timestamp;
      syncLiveInspectorVideo(false);
      if (state.inspector.imuPendingToken !== token && Math.abs(state.currentTimeMs - lastFetchTime) >= 1000) {
        lastFetchTime = state.currentTimeMs;
        loadRealImuSamples(state.inspector.carID, lastFetchTime);
      }
    }
    // The cursor and values follow the same clock as the car every frame;
    // fetching windows and correcting the native decoder can run less often.
    renderRealAccelerometer((state.currentTimeMs - (state.inspector.sampleCenterMs ?? state.currentTimeMs))/1000,
      state.inspector.severity);

    state.inspector.animId = requestAnimationFrame(liveTick);
  }

  state.inspector.animId = requestAnimationFrame(liveTick);
}

/**
 * Sync Live Inspector Video to current simulation time
 */
function syncLiveInspectorVideo(forceSeek = false) {
  const sync = state.syncManifest;
  const target = resolveFleetVideo(sync, state.selectedScenario, state.inspector.carID, state.currentTimeMs);
  const cfg = getVehicleConfig(state.inspector.carID);

  if (el.inspectorHudLeft) {
    el.inspectorHudLeft.textContent = `REC // ${cfg.label} // ${(target.session || 'SESSION 2').toUpperCase()}`;
  }
  if (el.inspectorHudRight) {
    el.inspectorHudRight.textContent = `${target.message || 'LIVE'}`;
  }

  const vid = el.inspectorVideo;
  if (!vid) return;

  const expectedUrl = target.url || `/demo_view/videos/session2.mp4`;
  // currentSrc may be empty until resource selection finishes. Compare the
  // assigned URL so a slow load is not restarted on every animation frame.
  const sourceChanged = vid.src !== new URL(expectedUrl, document.baseURI).href;

  if (forceSeek || sourceChanged) state.inspector.videoNeedsAlignment = true;
  if (sourceChanged) {
    vid.pause();
    vid.src = expectedUrl;
    vid.load();
  }

  const shouldPlay = state.isPlaying && target.state === 'playing';
  if (!shouldPlay && !vid.paused) vid.pause();
  if (vid.readyState < 1 || vid.seeking) return;

  const drift = target.currentTime - vid.currentTime;
  // Preserve the initial alignment through metadata loading. Paused scrubs
  // need frame-level accuracy; live playback tolerates only a small drift.
  if (state.inspector.videoNeedsAlignment || Math.abs(drift) > (shouldPlay ? 0.4 : 0.04)) {
    state.inspector.videoNeedsAlignment = false;
    vid.playbackRate = state.playbackSpeed;
    if (Math.abs(drift) > 0.02) {
      vid.currentTime = target.currentTime;
      return;
    }
  }

  // Correct small decoder drift smoothly while the simulation remains the
  // shared clock for the map, video target, and IMU sample cursor.
  const correction = shouldPlay && Math.abs(drift) > 0.08
    ? Math.max(-0.2, Math.min(0.2, drift * 0.5)) : 0;
  const rate = state.playbackSpeed * (1 + correction);
  if (Math.abs(vid.playbackRate - rate) > 0.01) vid.playbackRate = rate;
  if (shouldPlay && vid.paused) vid.play().catch(() => {});
}

/**
 * Render 3-Axis Accelerometer Telemetry from REAL database samples
 * (simulated_car_imu_samples table, 100Hz) across ±5.0s window
 */
function renderRealAccelerometer(relSec, severity) {
  const canvas = el.accelWaveformCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Background Grid Lines
  ctx.fillStyle = '#11151c';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = '#232936';
  ctx.lineWidth = 1;
  // Center baseline line
  ctx.beginPath();
  ctx.moveTo(0, h * 0.5);
  ctx.lineTo(w, h * 0.5);
  ctx.stroke();

  // Impact vertical marker at center (t = 0.0s)
  const impactX = w * 0.5;
  ctx.strokeStyle = 'rgba(211, 47, 47, 0.6)';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();
  ctx.moveTo(impactX, 0);
  ctx.lineTo(impactX, h);
  ctx.stroke();
  ctx.setLineDash([]);

  const samples = state.inspector.imuSamples;
  let curZ = 1.0;
  let curX = 0.0;
  let curY = 0.0;

  if (samples && samples.length > 0) {
    // --- REAL DATABASE SAMPLES (simulated_car_imu_samples) ---
    let lo=0, hi=samples.length-1;
    while(lo<hi){const mid=(lo+hi)>>1;if(samples[mid].rel_s<relSec)lo=mid+1;else hi=mid;}
    const cur = samples[lo];
    curZ = cur.gz !== undefined ? cur.gz : cur.accel_z / 9.80665;
    curX = cur.gx !== undefined ? cur.gx : cur.accel_x / 9.80665;
    curY = cur.gy !== undefined ? cur.gy : cur.accel_y / 9.80665;

    // Scale helper: maps g value [-0.5g, +2.5g] to canvas Y [h, 0]
    const getY = (g) => h - ((g + 0.5) / 3.0) * h;

    // Plot Z-Axis (Vertical Shock) - Red
    ctx.strokeStyle = '#d32f2f';
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i += 2) {
      const s = samples[i];
      const sx = ((s.rel_s + 5.0) / 10.0) * w;
      const sy = getY(s.gz !== undefined ? s.gz : s.accel_z / 9.80665);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();

    // Plot X-Axis (Lateral) - Blue
    ctx.strokeStyle = '#1976d2';
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i += 2) {
      const s = samples[i];
      const sx = ((s.rel_s + 5.0) / 10.0) * w;
      const sy = getY(s.gx !== undefined ? s.gx : s.accel_x / 9.80665);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();

    // Plot Y-Axis (Longitudinal) - Green
    ctx.strokeStyle = '#388e3c';
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    for (let i = 0; i < samples.length; i += 2) {
      const s = samples[i];
      const sx = ((s.rel_s + 5.0) / 10.0) * w;
      const sy = getY(s.gy !== undefined ? s.gy : s.accel_y / 9.80665);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
  } else {
    for (const [chip, axis] of [[el.chipAccelX,'X'],[el.chipAccelY,'Y'],[el.chipAccelZ,'Z']]) {
      if (chip) chip.textContent = `${axis}: —`;
    }
    ctx.fillStyle = '#aaa';
    ctx.fillText('Waiting for recorded IMU samples…', 12, 20);
    return;
  }

  // Update Telemetry Badges with real values
  if (el.chipAccelZ) {
    el.chipAccelZ.textContent = `Z: ${curZ >= 0 ? '+' : ''}${curZ.toFixed(2)}g`;
    const isShock = Math.abs(curZ - 1.0) > 0.35;
    el.chipAccelZ.style.background = isShock ? '#fde8e4' : '#e8f5e9';
    el.chipAccelZ.style.color = isShock ? '#c62828' : '#2e7d32';
  }
  if (el.chipAccelX) {
    el.chipAccelX.textContent = `X: ${curX >= 0 ? '+' : ''}${curX.toFixed(2)}g`;
  }
  if (el.chipAccelY) {
    el.chipAccelY.textContent = `Y: ${curY >= 0 ? '+' : ''}${curY.toFixed(2)}g`;
  }

  // Moving Playhead Cursor (0 to w across -5s to +5s)
  const needleX = ((relSec + 5.0) / 10.0) * w;
  ctx.strokeStyle = '#ffeb3b';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(needleX, 0);
  ctx.lineTo(needleX, h);
  ctx.stroke();

  // Playhead dot on Z-waveform
  const curY_px = h - ((curZ + 0.5) / 3.0) * h;
  ctx.fillStyle = '#ffeb3b';
  ctx.beginPath();
  ctx.arc(needleX, Math.max(3, Math.min(h - 3, curY_px)), 3.5, 0, Math.PI * 2); ctx.fill();
}

// Backward-compatible alias
const renderMockAccelerometer = renderRealAccelerometer;

/**
 * Animated Dashcam Simulation: draws interactive road dashcam
 */
function drawDashcamScene(ctx, width, height, dashOffset, carName, badgeColor) {
  const skyGrad = ctx.createLinearGradient(0, 0, 0, height * 0.45);
  skyGrad.addColorStop(0, '#1a2634');
  skyGrad.addColorStop(1, '#3b4d61');
  ctx.fillStyle = skyGrad;
  ctx.fillRect(0, 0, width, height * 0.45);

  ctx.fillStyle = '#142012';
  ctx.beginPath();
  ctx.moveTo(0, height * 0.45);
  for (let x = 0; x <= width; x += 20) {
    const y = height * 0.42 + Math.sin(x * 0.05) * 4;
    ctx.lineTo(x, y);
  }
  ctx.lineTo(width, height * 0.45);
  ctx.fill();

  const roadGrad = ctx.createLinearGradient(0, height * 0.45, 0, height);
  roadGrad.addColorStop(0, '#333333');
  roadGrad.addColorStop(1, '#1e1e1e');
  ctx.fillStyle = roadGrad;
  ctx.beginPath();
  ctx.moveTo(width * 0.4, height * 0.45);
  ctx.lineTo(width * 0.6, height * 0.45);
  ctx.lineTo(width * 0.95, height);
  ctx.lineTo(width * 0.05, height);
  ctx.closePath();
  ctx.fill();

  ctx.fillStyle = '#223820';
  ctx.beginPath();
  ctx.moveTo(0, height * 0.45);
  ctx.lineTo(width * 0.4, height * 0.45);
  ctx.lineTo(width * 0.05, height);
  ctx.lineTo(0, height);
  ctx.fill();

  ctx.beginPath();
  ctx.moveTo(width * 0.6, height * 0.45);
  ctx.lineTo(width, height * 0.45);
  ctx.lineTo(width, height);
  ctx.lineTo(width * 0.95, height);
  ctx.fill();

  ctx.fillStyle = '#ffdd44';
  const startY = height * 0.45;
  const totalH = height - startY;
  const numDashes = 6;

  for (let i = 0; i < numDashes; i++) {
    const progress = ((i * 35 + dashOffset) % 180) / 180;
    const y = startY + progress * totalH;
    const dashH = Math.max(3, progress * 16);
    const dashW = Math.max(2, progress * 6);
    const centerX = width * 0.5;
    ctx.fillRect(centerX - dashW / 2, y, dashW, dashH);
  }

  ctx.fillStyle = badgeColor || '#df6e35';
  ctx.beginPath();
  ctx.moveTo(width * 0.2, height);
  ctx.quadraticCurveTo(width * 0.5, height - 14, width * 0.8, height);
  ctx.fill();

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.font = '9px "Courier New", monospace';
  ctx.fillStyle = '#39ff14';
  ctx.fillText(`CAM // ${carName}`, 8, 14);
}

/**
 * RequestAnimationFrame Playback Loop
 */
function playbackLoop(timestamp) {
  if (!state.isPlaying) return;

  if (state.lastFrameTime === null) {
    state.lastFrameTime = timestamp;
  }

  const deltaSeconds = (timestamp - state.lastFrameTime) / 1000.0;
  state.lastFrameTime = timestamp;

  state.currentTimeMs += deltaSeconds * 1000 * state.playbackSpeed;

  if (state.currentTimeMs >= state.endTimeMs) {
    state.currentTimeMs = state.endTimeMs;
    state.isPlaying = false;
    if (el.playPauseText) el.playPauseText.textContent = '▶ Play Replay';
    updateMapToCurrentTime(true);
    return;
  }

  updateMapToCurrentTime(true);
  state.animationFrameId = requestAnimationFrame(playbackLoop);
}

/**
 * Toggle Play / Pause
 */
function togglePlayPause() {
  if (scenarioLoading) return;
  state.isPlaying = !state.isPlaying;
  if (state.isPlaying) {
    if (state.currentTimeMs >= state.endTimeMs) {
      state.currentTimeMs = state.startTimeMs;
      state.ingestedAnomalyIds.clear();
      invalidateSegmentCaches();
    }
    state.lastFrameTime = null;
    if (el.playPauseText) el.playPauseText.textContent = '⏸ Pause Replay';
    state.animationFrameId = requestAnimationFrame(playbackLoop);
  } else {
    navigationScenario?.stopAudio();
    if (el.playPauseText) el.playPauseText.textContent = '▶ Play Replay';
    if (state.animationFrameId) {
      cancelAnimationFrame(state.animationFrameId);
    }
  }
  if (state.inspector.isOpen && !state.inspector.isIncident) syncLiveInspectorVideo();
}

/**
 * Reset Replay
 */
function resetReplay() {
  navigationScenario?.stopAudio();
  state.isPlaying = false;
  if (el.playPauseText) el.playPauseText.textContent = '▶ Play Replay';
  if (state.animationFrameId) {
    cancelAnimationFrame(state.animationFrameId);
  }
  state.currentTimeMs = state.startTimeMs;
  state.ingestedAnomalyIds.clear();
  invalidateSegmentCaches();
  updateMapToCurrentTime();
}

/**
 * Jump to Next Anomaly
 */
function jumpToNextAnomaly() {
  if (scenarioLoading) return;
  if (navigationScenario) {
    navigationScenario.stopAudio();
    const next = navigationScenario.nextEvent(state.currentTimeMs / 1000);
    state.currentTimeMs = next === undefined ? 0 : Math.max(0, (next - 2) * 1000);
    updateMapToCurrentTime();
    return;
  }
  const allAnomalies = state.allObservations
    .filter((d) => d.imu_defect_detected && d.yolo_pothole_detected)
    .sort((a, b) => a.timestamp - b.timestamp);

  const next = allAnomalies.find((a) => a.timestamp > state.currentTimeMs + 200);

  if (next) {
    state.currentTimeMs = Math.max(state.startTimeMs, next.timestamp - 300);
    invalidateSegmentCaches();
    updateMapToCurrentTime();
    if (map) map.panTo([next.latitude, next.longitude]);
    openTelemetryInspectorForPoint(next.carID || next.car_id, next.timestamp, next.latitude, next.longitude, true, 'CRITICAL');
  } else if (allAnomalies.length > 0) {
    state.currentTimeMs = Math.max(state.startTimeMs, allAnomalies[0].timestamp - 300);
    invalidateSegmentCaches();
    updateMapToCurrentTime();
    if (map) map.panTo([allAnomalies[0].latitude, allAnomalies[0].longitude]);
    openTelemetryInspectorForPoint(allAnomalies[0].carID || allAnomalies[0].car_id, allAnomalies[0].timestamp, allAnomalies[0].latitude, allAnomalies[0].longitude, true, 'CRITICAL');
  }
}

/**
 * Populate slide-out Town Ledger (Database Inspector) tables
 */
function populateDrawerTables() {
  if (el.detectionsTableBody) {
    if (!state.allObservations || state.allObservations.length === 0) {
      el.detectionsTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No observations</td></tr>';
    } else {
      const rowsHtml = state.allObservations.slice(0, 80).map((d) => {
        const q = (d.road_quality || 'good').toLowerCase();
        let sensors = 'GPS Only';
        if (d.imu_defect_detected && d.yolo_pothole_detected) sensors = 'IMU + YOLO';
        else if (d.imu_defect_detected) sensors = 'IMU Shock';
        else if (d.yolo_pothole_detected) sensors = 'YOLO Vision';

        const carName = (d.carID || d.car_id || 'Car').replace('sim-waterloo-2to5-', '');
        const latStr = (d.latitude != null) ? Number(d.latitude).toFixed(4) : '0.0000';
        const lonStr = (d.longitude != null) ? Number(d.longitude).toFixed(4) : '0.0000';

        return `
          <tr>
            <td><strong>${carName}</strong></td>
            <td><span class="badge-quality ${q}">${q}</span></td>
            <td>${sensors}</td>
            <td>${latStr}, ${lonStr}</td>
          </tr>
        `;
      }).join('');
      el.detectionsTableBody.innerHTML = rowsHtml;
    }
  }

  if (el.potholesTableBody) {
    if (!state.potholes || state.potholes.length === 0) {
      el.potholesTableBody.innerHTML = '<tr><td colspan="4" style="text-align: center;">No registered potholes</td></tr>';
    } else {
      const rowsHtml = state.potholes.slice(0, 50).map((p) => {
        const sev = getSeverityLabel(p.severity);
        const conf = p.confidence ? `${Math.round(Number(p.confidence) * 100)}%` : '85%';
        const pLatStr = (p.latitude != null) ? Number(p.latitude).toFixed(4) : '0.0000';
        const pLonStr = (p.longitude != null) ? Number(p.longitude).toFixed(4) : '0.0000';
        const hits = p.hit_count || 1;

        return `
          <tr>
            <td><strong>${sev}</strong></td>
            <td><span style="font-weight: 800; color: #3b6138;">${conf}</span> (${hits}x)</td>
            <td>Multi-sensor</td>
            <td>${pLatStr}, ${pLonStr}</td>
          </tr>
        `;
      }).join('');
      el.potholesTableBody.innerHTML = rowsHtml;
    }
  }
}

function formatDuration(ms) {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function setupVideoElements() {
  const vid = el.inspectorVideo;
  const canvas = el.inspectorCanvas;
  if (!vid) return;

  vid.addEventListener('canplay', () => {
    vid.style.display = 'block';
    if (canvas) canvas.style.display = 'none';
  });
  vid.addEventListener('error', () => {
    vid.style.display = 'none';
    if (canvas) canvas.style.display = 'block';
  });
}

/**
 * Update playback speed across simulation and video hardware decoder
 */
function setPlaybackSpeed(newSpeed) {
  state.playbackSpeed = newSpeed;
  if (el.inspectorVideo) {
    el.inspectorVideo.playbackRate = newSpeed;
  }
  if (el.speedBtns) {
    el.speedBtns.forEach((b) => {
      const spd = parseFloat(b.dataset.speed) || 1.0;
      b.classList.toggle('active', spd === newSpeed);
    });
  }
}

function wireEvents() {
  if (el.btnPlayPause) el.btnPlayPause.addEventListener('click', togglePlayPause);
  if (el.btnReset) el.btnReset.addEventListener('click', resetReplay);
  if (el.btnNextAnomaly) el.btnNextAnomaly.addEventListener('click', jumpToNextAnomaly);

  // Inspector Buttons
  if (el.btnCloseInspector) {
    el.btnCloseInspector.addEventListener('click', closeTelemetryInspector);
  }
  if (el.btnReplayClip) {
    el.btnReplayClip.addEventListener('click', () => {
      state.inspector.isIncident = true;
      if (el.inspectorSubtitle) el.inspectorSubtitle.textContent = '±5.0s Impact Incident Window';
      prepareAndPlayIncidentClip();
    });
  }
  if (el.btnLiveFeed) {
    el.btnLiveFeed.addEventListener('click', () => {
      openTelemetryInspectorForCar(state.inspector.carID);
    });
  }

  // Anomaly Toast Click: opens inspector immediately!
  if (el.anomalyToast) {
    el.anomalyToast.style.cursor = 'pointer';
    el.anomalyToast.addEventListener('click', () => {
      openTelemetryInspectorForPoint(
        state.inspector.carID,
        state.currentTimeMs,
        null,
        null,
        true,
        'CRITICAL'
      );
    });
  }

  // Safe checks on potentially removed elements
  if (el.btnSeed) {
    el.btnSeed.addEventListener('click', () => {
      loadObservationsData();
      alert('🔄 Re-loaded observations from simulated_car_observations table!');
    });
  }

  if (el.scenarioFilter) {
    el.scenarioFilter.addEventListener('change', (e) => {
      state.selectedScenario = e.target.value;
      loadObservationsData();
    });
  }

  if (el.vehicleFilter) {
    el.vehicleFilter.addEventListener('change', (e) => {
      state.selectedVehicle = e.target.value;
      updateMapToCurrentTime();
    });
  }

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'SELECT') {
      e.preventDefault();
      togglePlayPause();
    } else if (e.code === 'ArrowRight') {
      jumpToNextAnomaly();
    } else if (e.code === 'Escape') {
      closeTelemetryInspector();
    }
  });

  if (el.timelineSlider) {
    el.timelineSlider.addEventListener('input', (e) => {
      if (scenarioLoading) return;
      navigationScenario?.stopAudio();
      const elapsed = parseInt(e.target.value, 10);
      state.currentTimeMs = state.startTimeMs + elapsed;
      invalidateSegmentCaches();
      updateMapToCurrentTime();
      if (state.inspector.isOpen && navigationScenario) closeTelemetryInspector();
      if (state.inspector.isOpen) {
        if (state.inspector.isIncident) {
          state.inspector.centerTimeMs = state.currentTimeMs;
          state.inspector.windowStartMs = state.currentTimeMs - 5000;
          state.inspector.windowEndMs = state.currentTimeMs + 5000;
          prepareAndPlayIncidentClip();
        } else {
          syncLiveInspectorVideo(true);
        }
      }
    });
  }

  if (el.speedBtns) {
    el.speedBtns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const spd = parseFloat(btn.dataset.speed) || 1.0;
        setPlaybackSpeed(spd);
      });
    });
  }

  if (el.toggleAutoIngest) {
    el.toggleAutoIngest.addEventListener('change', (e) => {
      state.autoIngest = e.target.checked;
    });
  }

  if (el.btnOpenDrawer && el.dbDrawer) {
    el.btnOpenDrawer.addEventListener('click', () => {
      el.dbDrawer.classList.add('open');
      if (el.drawerBackdrop) el.drawerBackdrop.style.display = 'block';
      populateDrawerTables();
    });
  }

  const closeDrawer = () => {
    if (el.dbDrawer) el.dbDrawer.classList.remove('open');
    if (el.drawerBackdrop) el.drawerBackdrop.style.display = 'none';
  };

  if (el.drawerCloseBtn) el.drawerCloseBtn.addEventListener('click', closeDrawer);
  if (el.drawerBackdrop) el.drawerBackdrop.addEventListener('click', closeDrawer);

  if (el.tabDetectionsBtn && el.tabPotholesBtn) {
    el.tabDetectionsBtn.addEventListener('click', () => {
      el.tabDetectionsBtn.classList.add('active');
      el.tabPotholesBtn.classList.remove('active');
      if (el.tabDetectionsContent) el.tabDetectionsContent.style.display = 'block';
      if (el.tabPotholesContent) el.tabPotholesContent.style.display = 'none';
    });

    el.tabPotholesBtn.addEventListener('click', () => {
      el.tabPotholesBtn.classList.add('active');
      el.tabDetectionsBtn.classList.remove('active');
      if (el.tabPotholesContent) el.tabPotholesContent.style.display = 'block';
      if (el.tabDetectionsContent) el.tabDetectionsContent.style.display = 'none';
    });
  }
}

// Bootstrap Application
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupVideoElements();
  wireEvents();
  loadObservationsData();
});
