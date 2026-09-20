/** Resolve a car's camera using the shared map timestamp (Unix milliseconds).
 * A gap means pause/hide footage, never advance the map clock to skip it.
 */
export function resolveFleetVideo(manifest, variant, carID, mapTimestampMs) {
  const car = manifest.variants[variant]?.cars.find(c => c.carID === carID);
  if (!car || !Number.isFinite(mapTimestampMs)) throw new Error('Invalid car or map timestamp');
  const elapsed = (mapTimestampMs - Date.parse(car.launch_timestamp)) / 1000;
  if (elapsed < 0) return {state: 'waiting'};
  if (elapsed > car.duration_s) return {state: 'finished'};
  const sourceUs = car.source_start.source_us + Math.round(elapsed * 1e6);
  const session = manifest.sessions.find(s => sourceUs >= s.origin_unix_us &&
    sourceUs < s.origin_unix_us + Math.round(s.duration_s * 1e6));
  if (!session) return {state: 'gap', reason: 'recording-gap'};
  const rawTime = (sourceUs - session.origin_unix_us) / 1e6 - session.video_offset_s;
  if (rawTime < 0 || rawTime >= session.video_duration_s)
    return {state: 'gap', reason: 'video-unavailable', session: session.session};
  // Map original variable frame PTS to the rendered constant frame rate video.
  const pts = session.frame_pts_s;
  let lo = 0, hi = pts.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (pts[mid] <= rawTime) lo = mid + 1; else hi = mid; }
  const index = Math.max(0, lo - 1);
  const next = pts[index + 1] ?? session.video_duration_s;
  const phase = Math.min(0.999999, Math.max(0, (rawTime - pts[index]) / (next - pts[index])));
  return {state: 'playing', session: session.session, url: session.url,
    rawVideoTimeS: rawTime, frameIndex: index, currentTime: (index + phase) / session.fps};
}
