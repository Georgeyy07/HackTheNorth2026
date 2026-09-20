# Fleet discoveries, voice warnings, and automatic rerouting

Open **http://localhost:8770/navigation-demo**. Select a scenario and click **Play with voice**. The user gesture enables browser speech. Playback defaults to 4×; switching to 1× demonstrates real demo-clock timing. Voice can be disabled; visible warnings remain. Restart or scrub to replay a section. Seeking itself does not speak old alerts.

## Trip and narrative

- A: Columbia Street West west of Phillip Street, requested coordinates **43.475800, -80.543800** (the saved geometry uses OSRM's snapped road position).
- B: MidCampus Drive entrance near University Avenue West, **43.475000, -80.528000**.
- Initial selected route: Columbia → Phillip → University → MidCampus, **1,799.2 m / 190.8 s** provider estimate.
- Alternative: Columbia → Albert → Hickory → Hazel → MidCampus, **1,763.5 m / 181.7 s** provider estimate.

The initial route is the selected route via Phillip, not a claim that it is the fastest. Without newly reported risks, the small time saving alone does not satisfy our switch threshold. The shared Columbia approach leaves time to change routes before the junction. The alternative uses real connected streets, including an unrecorded Albert/Hickory/Hazel section. That section's good surface is an explicitly **synthetic demonstration prior**, not an inference result.

Two scouts replay actual session 3 GPS patches, starting from source seconds 70 and 150. Their clock shifts make reports arrive ahead of the subject car. A third green survey marker uses simulated movement along the alternate route and is labeled as such. It illustrates the assumed-good corridor; its trajectory does not prove the road's condition.

Three location reports are drawn from the frozen YOLO results and original calibrated IMU ensemble. Each has both flags true in the same patch; nearby repeated patches are deduplicated. No new defect labels were invented. Their source provenance, timestamps, and input hashes are in `road_viewer/demo_data/navigation.json`.

| Demo time | Event |
|---|---|
| 0:11.75 | Scout A reports a hazard near the shared final approach. |
| 0:19.00 | Scout B reports a hazard on University Avenue. Auto-reroute now reduces known hazards ahead from 2 to 1 and saves about 9 seconds. |
| 0:22.00 | Scout A reports another University Avenue hazard, which the bypass avoids. |
| 2:10.75 | In warning-only mode, first warning fires at approximately 200 m along the active route. |
| 2:40.25 | In reroute mode, the shared final-approach hazard still causes a warning. |

The two modes have the same trip, reports, and clock. Warning mode keeps the selected route and issues three one-time hazard warnings. Reroute mode switches once and issues one final-approach warning. With all reports known, the selected route has three hazard regions and the alternative has one. GPS regions are not verified pothole counts or lane-specific positions.

The demo subject follows road geometry with simulated constant speed derived from each saved route's distance and duration; individual turns, congestion, and stops are not a traffic simulation. The provider ETA estimates are cached for deterministic offline replay and are not live traffic estimates.

## Reusable backend functionality

`route_planner/navigation.py` implements the decision policy. `POST /api/navigation/evaluate` exposes it through FastAPI. The request includes:

- `routes`: candidate provider routes with unique `id`, `[lat, lon]` geometry, positive `duration_s`, optional nonoverlapping `quality_regions` (`start_m`, `end_m`, `grade`, `available_s`, `provenance`).
- `current_route_id`, `location`, `speed_mps`, and the shared `now_s` clock.
- `hazards`: unique IDs, coordinates, severity 0–1, and report `available_s`.
- `warned_ids` and `last_reroute_s` maintained by the caller for this trip.

The response contains remaining ETA/risk metrics, each reachable alternative's eligibility, an optional `reroute` proposal, and warnings. The UI applies the proposed route ID, remembers alert IDs, and sets `last_reroute_s` when it accepts the switch. Evaluate when new fleet evidence arrives and as GPS advances; do not reset state on every update. The endpoint is stateless to keep separate vehicles/trips independent.

A caller supplies fresh route candidates from its routing provider or existing route backend. This API does not continuously poll Tiger or fetch new geometry on its own. The presentation uses two saved provider routes and applies this same policy automatically while replaying discoveries. Its backend computes the timeline sequentially, admitting evidence only at its availability time; there is no hardcoded “switch at 19 seconds” command.

Decisions require:

- Alternative connects within 10 m of the current route position and has the same destination (within 15 m).
- At least 30 m of shared forward road before divergence, rather than a last-second turn, backward route, or teleport to a nearby road.
- At least 40% reduction in remaining risk penalty and at least 15 seconds of utility improvement.
- Added ETA no more than **both 30 seconds and 10%** of current remaining ETA. A faster route is allowed.
- At least 30 seconds since the last switch, and at least 150 m remaining before destination.

Utility cost = remaining ETA + 45 seconds per severity-weighted hazard + 0.08 seconds per known bad-road metre + 0.025 seconds per known medium-road metre. These are **demo policy weights**, not a validated safety model. Unknown quality contributes no asserted good coverage; lack of reports is never presented as proof of a safe road.

Warnings require speed ≥ 0.5 m/s and a reported hazard on the active route within 25 m laterally and **0–200 m ahead along the route**. Behind-the-car and off-route reports are excluded. Nearby hazard locations are grouped within 25 m along the route. The caller's `warned_ids` suppress repeated announcements. Replay ticks are 250 ms, so threshold crossings occur just below 200 m. There is no promise of GPS/lane-level precision.

The presentation is isolated: it does not update the Tiger fleet tables, production pothole records, existing source scores, or the `/api/route` engine. The existing fleet presentation remains at `/`; it links to the new navigation view.

## Files and rebuilding

- `route_planner/navigation.py`: geometry, remaining-route scoring, warnings, rerouting gates.
- `road_viewer/navigation.py`: validated API and deterministic scenario replay.
- `road_viewer/static/navigation.{html,css,js}`: map, scouts, subject car, report timeline, voice and route UI.
- `road_viewer/demo_data/navigation.json`: portable fixture; no credentials, model weights, or videos required.
- `scripts/build_navigation_demo.py`: reproducible fixture builder using recorded source files and cached/downloaded provider routes.
- `route_planner/tests/test_navigation.py`: causal report gating, warning distances, continuous rerouting, detour/cooldown/connectivity constraints, API validation.

Example rebuild:

```bash
.venv/bin/python scripts/build_navigation_demo.py \
 --imu /home/origami/pothole/reports/local_imu_sessions_20260920 \
 --vision /home/origami/pothole/reports/fleet_vision_snapshot_20260920 \
 --route-cache /tmp/navigation-provider-cache \
 --output road_viewer/demo_data/navigation.json
```

Provider geometry is obtained through the [OSRM Route API](https://project-osrm.org/docs/v5.24.0/api/#route-service). Rebuilding against live map data may change geometry/ETA; retain the checked-in fixture for the reviewed presentation. Restart the server after rebuilding because the fixture and sequential replay are cached per process.

## Validation

The new navigation tests and existing replay API tests pass (10 tests, excluding the previously known unrelated pothole CRUD test). Browser checks exercise the route switch, the speech request containing “200 metres,” and rewind. The broader routing/directions suite has one existing failure in `test_no_nearby_potholes_still_fills_in_real_alternate_routes`: the unchanged fallback router returns one rather than two paths. The presentation uses the two verified cached routes and is not dependent on that fallback.
