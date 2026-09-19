# Roadscope: test-drive replay

For your own recordings, see **[CSV inference and upload](../docs/CSV_INFERENCE.md)**.
Run `python -m road_viewer.server --uploads artifacts/user_drives --port 8766`
to enable uploads, even without an existing demo export. Uploaded drives use
the shipped ordinal ensemble and display good / medium / bad quality with
Kalman-filtered disturbance alerts. The instructions below describe the original
regression replay.

Open **http://localhost:8765** while the server is running. This is a browser
replay of recorded sensor data and the frozen ensemble's timed predictions.
It does not retrain or run neural-network inference in the browser.

From the repository root, after installing `pip install -e ".[viewer]"`:

```bash
npm ci --prefix road_viewer
python -m road_viewer.server --export artifacts/test_drive_inference --port 8765
```

Supply a directory containing `manifest.json` and the exported session folders.
Alternatively set `ROAD_VIEWER_EXPORT` and use
`uvicorn road_viewer.server:create_app --factory --port 8765`. Optional alert
profile exports can be supplied with `--filters` or `ROAD_VIEWER_FILTERS`.
These artifacts are not checked into Git. `--host 0.0.0.0` allows access from
another device on the same reachable network; the default is localhost.
There is no authentication: use it as a local demo.

The existing demonstration export has nine recordings, a moving GPS marker, finalized roughness colors,
disturbance alerts, acceleration/gyroscope charts, and a prediction timeline.
Play or pause, scrub to any time, choose 1×–50× speed, jump to the next alert,
follow the car, or fit the traveled route. Space toggles playback and arrow keys
move five seconds. Playback pauses when the tab is hidden.

At the bottom of each plot, the time axis is relative to the TEST recording's
start. The UTC clock comes from the original recording. The car follows observed
GPS fixes without interpolating future locations. It disappears when GPS is
more than three seconds stale; inference and sensor plots continue.

Road colors use the latest **final** patch estimate assigned to a GPS fix.
Several patches can share one fix, so road segments have the location precision
of GPS, not 100 Hz. Hollow markers show the most recent provisional patches.
Coral markers show disturbance onsets at their target road locations, after
their alert availability times. Event lists show only alerts that have arrived.
Going backward clears later colors, alerts, scores, and car positions.

The side cards show the latest finalized prediction, with its target time. The
prediction chart additionally shows provisional estimates with dashed lines.
Tail patches remain provisional. The combined disturbance category is not a
dedicated pothole classifier. IRI categories use the existing project thresholds
2/4/6 m/km; they are not independent human road-quality annotations. Kaggle lacks
measured IRI; LiRA lacks disturbance annotations and gyroscope readings. These
limitations are shown alongside the replay and in its explanation dialog.

All 509,109 prepared sensor samples remain available to the viewer. The service
returns sensor values, including null missing inputs, separately from prediction
updates. The client reconstructs predictions using only updates with
`available_s <= playback_time`. Original inference files remain untouched in
the directory supplied with `--export`.

Files:

- `server.py`: FastAPI catalog, replay payloads, static assets, exports, and
  optional background CSV inference endpoints enabled with `--uploads`.
- `static/replay.js`: pure clock-driven prediction/event state and sensor/GPS
  lookup, independently tested without a browser.
- `static/app.js`: playback controls, map layers, and Canvas time-series plots.
- `static/index.html`, `static/style.css`: responsive interface.
- `tests/`: state-machine tests, API fidelity checks, and browser integration.

The map uses local [Leaflet 1.9.4](https://leafletjs.com/examples/quick-start/)
assets and browser-fetched [OpenStreetMap tiles](https://operations.osmfoundation.org/policies/tiles/),
with visible attribution and normal browser caching. Only street tiles require
internet; local sensor plots and geographic overlays continue if tiles fail.
There is no tile prefetch or bulk download. The tile URL is supplied by the
catalog endpoint if a different provider is needed later.

Tests from the project root:

```bash
python -m pytest road_viewer/tests/test_server.py -q
npm test --prefix road_viewer
cd road_viewer
PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npx playwright install chromium
VIEWER_URL=http://127.0.0.1:8765 PLAYWRIGHT_BROWSERS_PATH="$PWD/.browsers" npm run test:browser
```

Browser tests require the server to be running. They cover all nine drives to
their final sample, time gating, provisional updates, event onsets, play/pause,
speed changes, scrubbing backward, GPS dropouts, missing gyroscope data, mobile
layout, and a failed map-tile connection. Results and screenshots are written to
`road_viewer/test-results/`.

Python API tests use small temporary fixtures. Browser integration tests expect
the original nine-drive export with its known timing/GPS cases. The ZIP download
is optional: put an archive next to the export folder with the same basename
(`test_drive_inference.zip`), or that endpoint returns 404.

Exports made with the current `configs/timeline.json` already apply the fixed Kalman filter to final disturbance scores. The viewer labels these **Kalman + hysteresis**, uses their stored updates directly, and ignores old external comparison profiles for that export. Raw gyro plots are optional recorded context; the active four-input model never consumes gyro.
