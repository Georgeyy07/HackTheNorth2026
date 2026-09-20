// Shared presentation semantics: ordinal probabilities never imply physical IRI.
export function ordinalSession(session) { return session?.quality_mode === 'ordinal'; }
export function qualityValue(row, ordinal) {
  if (!row?.valid) return null;
  if (!ordinal) return row.iri_m_per_km ?? null;
  const q = row.quality_probability;
  return q?.length === 3 ? (q[1] + 2 * q[2]) * 50 : null;
}
export function qualityText(row, ordinal) {
  const value = qualityValue(row, ordinal);
  if (value == null) return 'Waiting';
  return ordinal ? `${['Good','Medium','Bad'][row.quality_grade]} · roughness score ${value.toFixed(0)}/100`
    : `${['Good','Medium','Bad','Terrible'][row.quality_grade]} · ${value.toFixed(2)} m/km`;
}
export function visionFrameAt(vision, imuTime) {
  if (!vision || !Number.isFinite(vision.video_offset_s)) return null;
  const t = imuTime - vision.video_offset_s;
  if (t < 0 || t >= vision.duration_s) return null;
  const frames = vision.frames;
  let low = 0, high = frames.length;
  while (low < high) { const mid = (low + high) >>> 1; if (frames[mid].video_time_s <= t) low = mid + 1; else high = mid; }
  const frame = frames[low - 1];
  return frame && t - frame.video_time_s < 1 / vision.fps + .01 ? frame : null;
}
