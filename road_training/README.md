# Current road model

Use **acceleration X/Y/Z + speed**. The current model keeps per-window instance normalization and a statistics branch for both IRI and localized-disturbance prediction. It accepts `[batch,1024,4]`; no gyro is required.

- [Four-input dataset and augmentation](acceleration_speed.py)
- [Encoder, instance normalization and statistics branch](instance_model.py)
- [Trained four-member ensemble and model card](../models/acceleration_speed/README.md)
- [Live inference with overlap consensus and fixed Kalman filtering](timeline_stream.py)
- [Kalman score filter and alert hysteresis](alert_filter.py)
- [Default postprocessing settings](../configs/timeline.json)
- [Training guide](../docs/TRAINING.md)
- [Experimental ordinal roughness model, PVS importer and controlled results](../docs/ORDINAL.md)
- [Full-drive inference export with timestamps and GPS](export_test_drives.py)

Run from the repository root:

```python
from road_training.checkpoints import load_teachers
from road_training.timeline_stream import RoadTimelineStream

ensemble = load_teachers('models/acceleration_speed/ensemble.json', device='cuda')
stream = RoadTimelineStream(ensemble)
rows = stream.push(samples, mask)  # [N,4]: accel_x, accel_y, accel_z, speed.
```

Inputs are raw SI measurements: acceleration includes gravity in m/s²; speed is m/s. Supply a matching boolean mask. The model handles normalization internally. See [input orientation and units](../docs/DATA.md).

The default filter is the earlier fixed probability-space Kalman filter: Q/R=3.2, onset=.70, offset=.50. It updates once per finalized 160-ms patch, after three-window consensus and a 320-ms finalization delay. Raw consensus scores remain in `original_probability`. It leaves IRI unchanged, resets between drives or invalid patches, and never treats repeated provisional revisions as new observations. Call `stream.reset()` at every drive boundary.

Legacy seven-channel storage, baselines and experiments remain available. `AccelerationSpeedDataset` and the export path remove gyro before the current model sees any input. See the [repository quickstart](../README.md) for training and replay commands.
