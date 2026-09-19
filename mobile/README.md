# RoughRoute mobile

```sh
npm ci
npm start
```

Open with Expo Go on a physical Android or iOS device, or use an SDK 57 development build. Maps/navigation remain in their existing tabs. Use **Motion** for IMU capture. A simulator cannot validate the physical sensor conventions.

## Record and export

1. Rigidly mount the phone with its screen facing the occupants (toward the car's rear). Select **Top up**, **Top to car left**, or **Top to car right**. Optional roll/pitch/yaw corrections support other mounting angles; these are right-hand rotations of the nominal vehicle vector around X, then Y, then Z, in degrees.
2. Start recording and grant motion/location permissions. The app requests 100 Hz accelerometer/gyro and 1 Hz GPS. Actual delivery rate depends on the device.
3. Optionally use **Calibrate parked tilt** while parked on level ground. It requires at least 3 seconds of filter data and 2 seconds of low angular rate, near-gravity acceleration, and fresh GPS speed below 0.3 m/s. The calibration uses VQF's gravity direction to correct mounting tilt; yaw must be set mechanically or manually. A stale sensor sample disables calibration. Recalibrate after moving the mount. Calibration applies to the current recording only.
4. Stop to save. Saved recordings remain available across launches. **Export session** opens the system share sheet with the complete JSONL file. No upload happens automatically.

Recording continues when switching to the navigation tab. The screen stays awake during capture. Backgrounding the app stops and saves the session: background sensor recording is not implemented. A new recording starts a new file and filter. Missing sensors/permissions and file-write failures are surfaced in the UI.

## Raw and VQF data

Files live in the app's documents directory under `motion-recordings/`. Each UTF-8 line is a JSON object with a `type`:

| Type | Contents |
| --- | --- |
| `header` | Schema/version, platform, units, axes, initial device-to-vehicle rotation and timing/filter conventions |
| `accelerometer` | Original Expo `event` (XYZ in g and monotonic timestamp in seconds), Unix `receivedAt` in ms, validation flag; `si` is device XYZ in m/s² with gravity and positive specific-force convention |
| `gyroscope` | Original Expo `event` (XYZ in rad/s and monotonic timestamp in seconds), Unix receipt time and validation flag |
| `location` | Original Expo location fix, including timestamp, raw speed (m/s), coordinates/accuracy and receipt time; invalid native speed is retained here |
| `stabilized` | Synchronized 100 Hz source `time` (seconds), Unix `availableAt` (ms), segment, settling flag, quaternion, `earthAccel`, `earthGyro`, `verticalLinear`, `speedMps`, GPS `speedTimestamp`, vehicle `input`, `gyro`, and four-channel `mask` |
| `mount` | New fixed mount rotation and segment after parked calibration |
| `end` | Stop reason, counts and rejected fusion-event count |

All three accelerometer and gyro axes are recorded in **both** raw and VQF streams. Speed is recorded unchanged from GPS and included in synchronized rows; VQF does not estimate or smooth speed. Raw events are saved before validation/resampling, including duplicates and invalid events (`NaN`/infinity serialize to JSON `null`).

`earthAccel` is VQF-rotated acceleration **including gravity**, in m/s². `earthGyro` is the corresponding rotated gyro in rad/s. These use a Z-up earth frame with **arbitrary yaw**, not north/east and not car-forward/left. `quaternion` is scalar-first `[w,x,y,z]`, device-to-earth. `verticalLinear = earthAccel[2] - 9.80665` is only a separate gravity-free diagnostic.

Independent IMU streams are linearly interpolated onto a common 100 Hz source-time grid, requiring bracketing samples from both sensors. No stale extrapolation is performed. Gaps over 50 ms or excessive unmatched events reset fusion and start a new segment. Invalid/non-increasing source timestamps are excluded from fusion. The first 10 seconds of each segment are marked `settling`; those data are saved, not discarded. UI readings expire after 250 ms without a fused sample. Samples use the latest nonnegative finite GPS speed available at processing time for at most 3 seconds; missing speed is `null`, with a false mask. Speed is availability-aligned, not precise source-time interpolation, so both timestamps are retained.

Records are appended throughout capture, flushing every 200 ms or 32 KiB. Capture length is not limited by the 1,024-sample rolling preview buffer. Disk space is the practical limit. A crash may lose buffered records or leave a partial final line; a file without an `end` record is incomplete, but earlier complete lines remain usable. Do not concatenate different segments into one uninterrupted model window.

## Vehicle-relative model contract

`stabilized.input` always contains exactly:

```text
[accel_x, accel_y, accel_z, speed]
```

X is forward, Y left and Z up **with the car**, acceleration is m/s² including gravity, speed is m/s. Nominal upright Android mapping is `[-phone_z, -phone_x, phone_y]`. Stationary level input is approximately `[0, 0, +9.81, 0]`. Expo acceleration in g is converted with 9.80665; iOS Core Motion acceleration is additionally sign-flipped to the Android positive-specific-force convention. Gyro is already rad/s and is not sign-flipped.

The fixed mount rotation, including any parked VQF calibration, transforms **both** acceleration and gyro. It stays fixed as the vehicle turns, pitches, and rolls. Feeding continuously earth-levelled VQF XYZ into the model would violate the vehicle-body contract, so the VQF earth-frame streams are recorded separately. No gravity subtraction, acceleration smoothing or instance normalization is applied to `input`. Normalization belongs inside the model. Mount augmentation only covers modest offsets (about ±20° yaw and ±10° tilt), not arbitrary phone orientations. Dataset horizontal alignment, including MIT/UMass, is not verified by this recorder.

This branch has no live model endpoint or model weights in the mobile app. The recorder produces model-ready samples via `MotionPipeline.onSample` and records them for downstream inference; it does not claim to run predictions. The training archive's seven-channel schema is separate from this four-channel mobile contract.

## Filter and verification

The portable JavaScript implementation is the **BasicVQF 2.1.2 6D** algorithm: gyro integration, inertial-frame second-order Butterworth acceleration filtering (`tauAcc=3 s`) and inclination correction. It works in Expo Go without a custom native bridge. It is the upstream basic variant, **not** full VQF's rest/motion gyro-bias estimator or magnetic-disturbance rejection. No absolute heading is estimated. Vehicle yaw cannot be inferred from gravity alone.

Sources: [upstream VQF](https://github.com/dlaidig/vqf/tree/v2.1.2), [BasicVQF API](https://vqf.readthedocs.io/en/stable/ref_cpp_basic.html), [Expo SDK 57 accelerometer](https://docs.expo.dev/versions/v57.0.0/sdk/accelerometer/), [gyroscope](https://docs.expo.dev/versions/v57.0.0/sdk/gyroscope/). Upstream attribution/license is in `src/motion/VQF-LICENSE.txt`.

```sh
npm test
npx expo export --platform android --platform ios
```

Tests compare the port to an independently compiled upstream C++ fixture over 4,000 updates, including filter initialization, motion and reset. They also cover mapping/signs, side/tilted mounts, gravity retention, interpolation/gaps, stale GPS, parked calibration and recording beyond the preview buffer. See `tests/reference/generate_vqf.cpp` to regenerate the fixture with upstream v2.1.2 sources.

Before a road trial, verify on real hardware: upright rest gives vehicle Z ≈ +9.81; controlled forward/left/up impulses have the expected signs; a sideways/tilted mount corrects as configured; turning the whole mount preserves vehicle axes; start/stop/background retains files and export includes both raw and fused records. Bundling/unit tests do not replace that hardware check.
