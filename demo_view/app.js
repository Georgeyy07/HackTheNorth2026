/**
 * RoadScope Pitch Demo View — Telemetry Playback & Road Quality Ledger
 * Built on `simulated_car_observations` table in Tiger PostgreSQL Cloud.
 * Implements 4-car scenario playback, color-coded road quality trails,
 * 15m spatial anomaly merging with confidence boosting,
 * and on-demand Dashcam Video & Accelerometer (IMU) Telemetry Inspection.
 */

import { resolveFleetVideo } from './fleet-video-sync.js';

const API_BASE = (window.location.origin && window.location.origin.includes('8765'))
  ? window.location.origin
  : 'http://localhost:8765';

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
  try {
    if (el.dbStatusText) el.dbStatusText.textContent = 'Connecting...';

    // 1. Fetch Fleet Video Sync manifest
    try {
      const syncRes = await fetch(`${API_BASE}/api/fleet-sync`);
      if (syncRes.ok) {
        state.syncManifest = await syncRes.json();
      }
    } catch (e) {
      console.warn('Could not load fleet-sync manifest:', e);
    }

    // 2. Fetch observations from simulated_car_observations table
    const obsUrl = `${API_BASE}/api/simulated-car-observations?scenario=${encodeURIComponent(state.selectedScenario)}&limit=50000`;
    const obsRes = await fetch(obsUrl);
    if (!obsRes.ok) throw new Error(`HTTP ${obsRes.status}`);
    const rawObs = await obsRes.json();
    state.allObservations = rawObs;

    // 3. Fetch active potholes from central database
    try {
      const potholesRes = await fetch(`${API_BASE}/api/potholes`);
      if (potholesRes.ok) {
        const rawPotholes = await potholesRes.json();
        // Client-side 15m merge pass to eliminate duplicates & boost confidences
        state.potholes = mergeDuplicatePotholes(rawPotholes, 15.0);
      }
    } catch (e) {
      console.warn('Could not load potholes:', e);
    }

    if (el.dbStatusText) el.dbStatusText.textContent = 'TigerDB (simulated_car_observations)';
    processObservations();
    renderPotholes();
    populateDrawerTables();
  } catch (err) {
    console.error('Failed to load observations data:', err);
    if (el.dbStatusText) el.dbStatusText.textContent = 'Error loading observations';
  }
}

/**
 * Group observations by carID, sort chronologically, and create
 * persistent, high-performance polyline segments (zero lag during playback).
 */
function processObservations() {
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

        trailLayer.addLayer(polyline);

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
function updateMapToCurrentTime() {
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
      segments.forEach((seg) => {
        if (seg.isFull || seg.polyline.getLatLngs().length > 0) {
          seg.polyline.setLatLngs([]);
          seg.isFull = false;
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

      // Check anomaly detection with 15m spatial debouncing
      if (currWp.imu_defect_detected && currWp.yolo_pothole_detected) {
        const lastPos = state.lastAnomalyPosPerVehicle.get(carID);
        const distFromLast = lastPos ? getDistanceMeters(currWp.latitude, currWp.longitude, lastPos.lat, lastPos.lon) : Infinity;
        if (distFromLast >= 15.0) {
          state.lastAnomalyPosPerVehicle.set(carID, { lat: currWp.latitude, lon: currWp.longitude });
          handleAnomalyEncountered(currWp, carID);
        }
      }
    } else {
      currentPos = [waypoints[0].latitude, waypoints[0].longitude];
    }

    // Update car marker position
    if (marker && currentPos) {
      marker.setLatLng(currentPos);
    }

    // Update persistent polyline segments (0 recreation, 0 DOM thrashing)
    segments.forEach((seg) => {
      if (seg.startTs > state.currentTimeMs) {
        // Future segment
        if (seg.isFull || seg.polyline.getLatLngs().length > 0) {
          seg.polyline.setLatLngs([]);
          seg.isFull = false;
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
        }
      }
    });
  });

  // Update Scrubber UI
  const elapsedMs = state.currentTimeMs - state.startTimeMs;
  if (el.timelineSlider) el.timelineSlider.value = Math.max(0, elapsedMs);
  if (el.currentTimeLabel) el.currentTimeLabel.textContent = formatDuration(elapsedMs);

  // If inspector is open and tracking live car, sync it
  if (state.inspector.isOpen && !state.inspector.isIncident) {
    syncLiveInspectorVideo(false);
  }
}

/**
 * Invalidate cached full polylines on scrub / jump
 */
function invalidateSegmentCaches() {
  state.vehicleSegments.forEach((segments) => {
    segments.forEach((seg) => {
      seg.isFull = false;
      seg.lastWpIdx = -1;
    });
  });
}

/**
 * Handle vehicle encountering an anomaly point during replay:
 * 15m SPATIAL MERGING: If a pothole exists within 15m, merge and boost confidence!
 */
async function handleAnomalyEncountered(wp, carID) {
  const imu = Boolean(wp.imu_defect_detected || wp.imu);
  const yolo = Boolean(wp.yolo_pothole_detected || wp.yolo);
  const sevLabel = (imu && yolo) ? 'CRITICAL' : 'HIGH';

  // Check if an existing pothole is within 15 meters
  const nearby = findNearbyPothole(wp.latitude, wp.longitude, 15.0);

  if (nearby) {
    // 15m SPATIAL MERGE!
    const oldConf = Number(nearby.confidence) || 0.85;
    const boostedConf = Math.min(0.99, Math.round((oldConf + 0.05) * 100) / 100);
    nearby.confidence = boostedConf;
    nearby.hit_count = (nearby.hit_count || 1) + 1;
    if (sevLabel === 'CRITICAL') nearby.severity = 'CRITICAL';

    // Visual pulse & update on map marker
    updatePotholeMarker(nearby, true);
    populateDrawerTables();

    // Show Toast
    if (el.toastCarId) el.toastCarId.textContent = carID;
    if (el.toastDesc) el.toastDesc.textContent = `Pothole Re-confirmed! Merged with existing defect within 15m. Confidence boosted to ${Math.round(boostedConf * 100)}%.`;
    if (el.toastImuPill) {
      el.toastImuPill.textContent = 'IMU Shock: DETECTED';
      el.toastImuPill.style.background = '#fde8e4';
      el.toastImuPill.style.color = '#b83824';
    }
    if (el.toastYoloPill) {
      el.toastYoloPill.textContent = 'YOLO Vision: CONFIRMED';
      el.toastYoloPill.style.background = '#fdf3e4';
      el.toastYoloPill.style.color = '#d4652c';
    }
    if (el.toastIngestStatus) {
      el.toastIngestStatus.innerHTML = `<span>🔄 Merged into Pothole #${nearby.id || 'canonical'} (Confidence: ${Math.round(boostedConf * 100)}%)</span>`;
    }
    if (el.anomalyToast) el.anomalyToast.style.display = 'block';

    // Sync merge to PostgreSQL TigerDB
    try {
      await fetch(`${API_BASE}/api/potholes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: wp.latitude,
          longitude: wp.longitude,
          severity: nearby.severity,
          confidence: boostedConf,
          detected_by_vision: yolo,
          detected_by_imu: imu,
        }),
      });
    } catch (e) {
      console.warn('Sync merge error:', e);
    }
  } else {
    // Brand new pothole (> 15m from any existing)
    const newPothole = {
      id: `pothole-${Date.now()}-${Math.floor(Math.random() * 1000)}`,
      latitude: wp.latitude,
      longitude: wp.longitude,
      severity: sevLabel,
      confidence: 0.85,
      hit_count: 1,
      timestamp: new Date().toISOString(),
      detected_by_vision: yolo,
      detected_by_imu: imu,
    };
    state.potholes.push(newPothole);
    updatePotholeMarker(newPothole, true);
    if (el.statPotholes) el.statPotholes.textContent = state.potholes.length;
    populateDrawerTables();

    // Show Toast
    if (el.toastCarId) el.toastCarId.textContent = carID;
    if (el.toastDesc) el.toastDesc.textContent = `New road anomaly cataloged at (${wp.latitude.toFixed(4)}, ${wp.longitude.toFixed(4)})`;
    if (el.toastImuPill) {
      el.toastImuPill.textContent = `IMU Shock: ${imu ? 'DETECTED' : 'Normal'}`;
      el.toastImuPill.style.background = imu ? '#fde8e4' : '#e1ecd9';
      el.toastImuPill.style.color = imu ? '#b83824' : '#3b6138';
    }
    if (el.toastYoloPill) {
      el.toastYoloPill.textContent = `YOLO Vision: ${yolo ? 'CONFIRMED' : 'Normal'}`;
      el.toastYoloPill.style.background = yolo ? '#fdf3e4' : '#e1ecd9';
      el.toastYoloPill.style.color = yolo ? '#d4652c' : '#3b6138';
    }
    if (el.toastIngestStatus) {
      el.toastIngestStatus.innerHTML = `<span>✍️ Cataloging into central database...</span>`;
    }
    if (el.anomalyToast) el.anomalyToast.style.display = 'block';

    // Sync to PostgreSQL TigerDB
    try {
      const resp = await fetch(`${API_BASE}/api/potholes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          latitude: wp.latitude,
          longitude: wp.longitude,
          severity: sevLabel,
          confidence: 0.85,
          detected_by_vision: yolo,
          detected_by_imu: imu,
        }),
      });
      if (resp.ok) {
        const saved = await resp.json();
        if (saved.id) newPothole.id = saved.id;
        if (saved.confidence) newPothole.confidence = saved.confidence;
        updatePotholeMarker(newPothole, false);
        if (el.toastIngestStatus) {
          el.toastIngestStatus.innerHTML = `<span>✅ Live Seeded into Database! (ID: ${newPothole.id})</span>`;
        }
      }
    } catch (e) {
      console.warn('Sync create error:', e);
    }
  }

  setTimeout(() => {
    if (el.anomalyToast && el.anomalyToast.style.display === 'block') {
      el.anomalyToast.style.display = 'none';
    }
  }, 3500);
}

/* ==========================================================================
   Telemetry & Dashcam Inspector (±5.0s Video & 3-Axis Mock Telemetry)
   ========================================================================== */

/**
 * Stop any running inspector animation loop and pause video
 */
function stopInspectorLoop() {
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
function closeTelemetryInspector() {
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
  try {
    const resp = await fetch(`${API_BASE}/api/imu-samples?car_id=${encodeURIComponent(carID)}&timestamp=${centerTimeMs}&window_seconds=5.0`);
    if (resp.ok) {
      const data = await resp.json();
      if (data && Array.isArray(data.samples) && data.samples.length > 0) {
        state.inspector.imuSamples = data.samples;
        renderRealAccelerometer(state.inspector.relativeSec, state.inspector.severity);
        return;
      }
    }
  } catch (e) {
    console.warn('Failed to load real IMU samples from simulated_car_imu_samples:', e);
  }
  state.inspector.imuSamples = [];
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
  loadRealImuSamples(state.inspector.carID, state.inspector.centerTimeMs);

  // Compute start video position from sync manifest (-5.0s from impact center)
  const sync = state.syncManifest;
  const startTarget = resolveFleetVideo(sync, state.selectedScenario, state.inspector.carID, state.inspector.windowStartMs);
  const expectedUrl = startTarget.url || `/demo_view/videos/session2.mp4`;
  const targetStartTime = Math.max(0, startTarget.currentTime);

  state.inspector.clipStartVideoTime = targetStartTime;
  state.inspector.clipEndVideoTime = targetStartTime + 10.0;

  const cfg = getVehicleConfig(state.inspector.carID);
  if (el.inspectorHudLeft) {
    el.inspectorHudLeft.textContent = `REC // ${cfg.label} // ${(startTarget.session || 'SESSION 2').toUpperCase()}`;
  }

  // Callback once the video has reached the target frame and is ready
  const onVideoReady = () => {
    if (token !== state.inspector.loadToken || !state.inspector.isOpen || !state.inspector.isIncident) return;

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
      vid.addEventListener('seeked', onSeeked, { once: true });
      vid.currentTime = targetStartTime;

      // Fallback timeout in case seeked event doesn't fire
      setTimeout(() => {
        if (!fired && token === state.inspector.loadToken && state.inspector.isOpen) {
          fired = true;
          vid.removeEventListener('seeked', onSeeked);
          onVideoReady();
        }
      }, 1500);
    }
  };

  const expectedFileName = expectedUrl.split('/').pop();
  const currentFileName = vid.currentSrc ? vid.currentSrc.split('/').pop() : '';

  if (currentFileName !== expectedFileName) {
    // Switch video file source
    vid.pause();
    vid.addEventListener('loadedmetadata', () => {
      if (token !== state.inspector.loadToken) return;
      seekToStart();
    }, { once: true });
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
      }, { once: true });
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
function openTelemetryInspectorForPoint(carID, timestamp, lat, lon, isIncident = true, sev = 'CRITICAL') {
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

  syncLiveInspectorVideo(true);

  function liveTick() {
    if (token !== state.inspector.loadToken || !state.inspector.isOpen || state.inspector.isIncident) {
      return;
    }

    syncLiveInspectorVideo(false);

    // Live accelerometer based on current waypoint quality
    renderMockAccelerometer(0.0, state.inspector.severity);

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
  const expectedFileName = expectedUrl.split('/').pop();
  const currentFileName = vid.currentSrc ? vid.currentSrc.split('/').pop() : '';

  if (currentFileName !== expectedFileName) {
    vid.src = expectedUrl;
    vid.load();
    vid.playbackRate = state.playbackSpeed;
    if (vid.readyState >= 1) {
      vid.currentTime = target.currentTime;
    }
    return;
  }

  if (vid.playbackRate !== state.playbackSpeed) {
    vid.playbackRate = state.playbackSpeed;
  }

  if (vid.readyState >= 1) {
    const drift = Math.abs(vid.currentTime - target.currentTime);
    // Only seek if forced (scrub) or if drift is major (> 2.5s) to avoid decoder thrashing
    if (forceSeek || drift > 2.5) {
      vid.currentTime = target.currentTime;
    }

    if (state.isPlaying) {
      if (vid.paused) vid.play().catch(() => { });
    } else {
      if (!vid.paused) vid.pause();
    }
  }
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
    const pct = Math.max(0, Math.min(1, (relSec + 5.0) / 10.0));
    const approxIdx = Math.max(0, Math.min(samples.length - 1, Math.round(pct * (samples.length - 1))));
    const cur = samples[approxIdx];
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
    // Fallback while database samples are loading
    const dt = relSec;
    curZ = 1.0 + Math.sin(dt * 12) * 0.07;
    curX = Math.cos(dt * 7) * 0.04;
    curY = Math.sin(dt * 5) * 0.04;
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
    updateMapToCurrentTime();
    return;
  }

  updateMapToCurrentTime();
  state.animationFrameId = requestAnimationFrame(playbackLoop);
}

/**
 * Toggle Play / Pause
 */
function togglePlayPause() {
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
    if (el.playPauseText) el.playPauseText.textContent = '▶ Play Replay';
    if (state.animationFrameId) {
      cancelAnimationFrame(state.animationFrameId);
    }
  }
}

/**
 * Reset Replay
 */
function resetReplay() {
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
      const elapsed = parseInt(e.target.value, 10);
      state.currentTimeMs = state.startTimeMs + elapsed;
      invalidateSegmentCaches();
      updateMapToCurrentTime();
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
