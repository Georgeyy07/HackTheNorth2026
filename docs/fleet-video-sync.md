# Fleet video synchronization (sessions 2–5)

Session 1 is excluded from the active fleet database bundle and video playlist. The scenario bundle is `/home/origami/pothole/reports/fleet_sessions_2_5`. Keep its `video_sync.json` alongside the database export: the seven database columns alone do not identify source video time.

Fetch `http://localhost:8770/api/fleet-sync`. Import `resolveFleetVideo` from `/static/fleet-video-sync.js`. Pass the selected variant (`staggered` or `cascade`), exact database `carID`, and **the map's simulated Unix timestamp in milliseconds**, not wall-clock time:

```js
import {resolveFleetVideo} from '/static/fleet-video-sync.js';
const sync = await (await fetch('/api/fleet-sync')).json();
const target = resolveFleetVideo(sync, variant, carID, mapTimestampMs);
// For target.state === 'playing': select target.url and seek to target.currentTime.
// Otherwise pause and show waiting / recording gap / finished, without stale footage.
```

The map starts at `2026-09-20T12:00:00Z`. If using its relative timeline, `mapTimestampMs = Date.parse(sync.start) + mapTimeSeconds * 1000`. For each car:

```
sourceAbsoluteUs = sourceStartUs + (mapTimestampMs - launchTimestampMs) * 1000
sessionSeconds = (sourceAbsoluteUs - sessionOriginUs) / 1e6
originalVideoSeconds = sessionSeconds - videoOffsetSeconds
```

The helper selects the source session automatically and converts original frame timestamps to the annotated video's constant-frame-rate timestamps. Video endpoints support range requests. Deploy the same annotated files behind the supplied URLs (or rewrite URLs if the video host differs).

## Simultaneous starts

All four cars launch at 12:00:00 UTC. IDs are `sim-waterloo-2to5-staggered-01` through `04`.

| Car | First session | IMU start seconds | Video at launch |
|---|---|---:|---|
| 01 | session2 | 9.280 | 8.902 seconds |
| 02 | session3 | 0.320 | First frame after 0.118 seconds |
| 03 | session4 | 0.320 | First frame after 0.043 seconds |
| 04 | session5 | 0.320 | First frame after 0.294 seconds |

For cascade IDs `sim-waterloo-2to5-cascade-01` through `04`, launches occur at 12:00:00, 12:00:20, 12:00:40, and 12:01:00 UTC. Each starts session2 video at 8.902 seconds, then follows sessions3–5. Do not play footage before that car launches.

Use one shared map clock for play, pause, speed, restart and scrubbing. On media changes wait for loaded metadata before seeking; ignore stale load callbacks after a seek or scenario change. While playing, match the map playback rate and correct drift against the helper. Pause all videos when the map pauses. A seek must resolve the video again, including backward seeks across sessions. Avoid independent per-car elapsed timers.

Recording gaps between sessions are preserved (1.610, 5.615 and 3.919 seconds), as are uncovered video heads/tails. Do not concatenate clips and remove these gaps. Missing GPS observations need not stop the video clock; they only stop new map observations. At gaps, show an unavailable state instead of frozen footage that appears current.

Sessions2–5 are aligned as confirmed by the recording owner; subsecond offsets (.378, .438, .363, .614 seconds) retain the existing MP4-metadata convention used by the YOLO database import. This is consistent alignment with the imported detections, not an independently measured frame-accurate synchronization calibration.

**Label timing:** annotated videos have YOLO boxes on their corresponding camera frames. Their burned-in IMU labels use delayed inference availability; database/map labels use retrospective target patch ends. Overlay the database labels in the player if matching map labels exactly is required. Do not shift the whole video to compensate for inference latency.

Regenerate the sidecar with `python scripts/build_fleet_video_sync.py BUNDLE RENDERED_DIRECTORY`. It verifies source video hashes, checkpoint identity, frame count, and frame timestamps against the frozen YOLO metadata used by the fleet import.
