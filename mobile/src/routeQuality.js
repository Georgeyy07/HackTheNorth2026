// Colours a route by the road quality along it, so the map shows *where* a
// path is rough rather than only reporting a count for the whole trip.

export const QUALITY_COLORS = {
  clean: '#2f7d32',     // green  -- nothing reported on this stretch
  moderate: '#d79a1e',  // yellow -- a pothole or two, or a minor one
  rough: '#c1352b',     // red    -- several, or a severe one
};

// A CRITICAL pothole makes a stretch rough on its own; a single LOW one only
// nudges it to moderate. Anything unrecognised is treated as a mild hit
// rather than ignored, since severity labels have been renamed before.
const SEVERITY_WEIGHT = { LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4 };
const DEFAULT_WEIGHT = 1;

const MODERATE_AT = 1;
const ROUGH_AT = 3;

export function severityWeight(severity) {
  return SEVERITY_WEIGHT[String(severity).toUpperCase()] ?? DEFAULT_WEIGHT;
}

export function scoreColor(score) {
  if (score >= ROUGH_AT) return QUALITY_COLORS.rough;
  if (score >= MODERATE_AT) return QUALITY_COLORS.moderate;
  return QUALITY_COLORS.clean;
}

function nearestIndex(coords, lat, lon) {
  // Longitude degrees shrink toward the poles; scaling by cos(lat) keeps the
  // comparison honest without the cost of a real distance for every point.
  const cosLat = Math.cos((lat * Math.PI) / 180);
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < coords.length; i++) {
    const dy = lat - coords[i][0];
    const dx = (lon - coords[i][1]) * cosLat;
    const dist = dy * dy + dx * dx;
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

/**
 * Splits `coords` ([[lat, lon], ...]) into consecutive runs, each carrying the
 * colour earned by the potholes nearest it.
 *
 * Runs share a point with the next one so the drawn line stays unbroken --
 * without that overlap the map shows a gap at every colour change.
 */
export function buildQualitySegments(coords, potholes = [], pointsPerSegment = 10) {
  if (!Array.isArray(coords) || coords.length < 2) return [];
  const size = Math.max(1, pointsPerSegment);
  const segmentCount = Math.ceil((coords.length - 1) / size);

  const scores = new Array(segmentCount).fill(0);
  for (const pothole of potholes) {
    const lat = pothole?.lat;
    const lon = pothole?.lon;
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
    const index = nearestIndex(coords, lat, lon);
    // The final point belongs to the last segment, not a segment past the end.
    const segment = Math.min(segmentCount - 1, Math.floor(index / size));
    scores[segment] += severityWeight(pothole.severity);
  }

  const segments = [];
  for (let s = 0; s < segmentCount; s++) {
    const start = s * size;
    const end = Math.min(coords.length - 1, start + size);
    segments.push({
      coordinates: coords.slice(start, end + 1).map(([lat, lon]) => ({ latitude: lat, longitude: lon })),
      color: scoreColor(scores[s]),
      score: scores[s],
    });
  }
  return segments;
}
