// IDs identify retries; they are not credentials or security tokens.
export const frameId = () => 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
  const r = Math.floor(Math.random() * 16);
  return (c === 'x' ? r : (r & 3) | 8).toString(16);
});

export function cameraFrame({ image, id, deviceId, number, timestamp, position }) {
  if (!image) throw new Error('Capture a fresh image before sending a frame.');
  const gps = position?.coords;
  const valid = Number.isFinite(position?.timestamp) && Number.isFinite(position?.receivedAt)
    && timestamp >= position.timestamp && timestamp-position.timestamp <= 3000
    && position.receivedAt <= timestamp;
  return {
    type: 'camera_frame', frame_id: id, device_id: deviceId, frame_number: number,
    timestamp, image,
    latitude: valid ? gps?.latitude ?? null : null,
    longitude: valid ? gps?.longitude ?? null : null,
    gps_timestamp_ms: valid ? position.timestamp : null,
    gps_received_at_ms: valid ? position.receivedAt : null,
  };
}
