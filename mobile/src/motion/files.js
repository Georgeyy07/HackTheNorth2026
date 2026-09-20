import { Directory, File, FileMode, Paths } from 'expo-file-system';

const recordingsDirectory = () => new Directory(Paths.document, 'motion-recordings');

/** Shared basename for a session's JSONL and its paired demonstration video. */
export function recordingTag() {
  return `${new Date().toISOString().replace(/[:.]/g, '-')}-${Math.random().toString(36).slice(2, 8)}`;
}

export function createRecordingSink(tag = recordingTag()) {
  const directory = recordingsDirectory();
  directory.create({ idempotent: true, intermediates: true });
  const name = `imu-${tag}.jsonl`;
  const file = new File(directory, name);
  file.create();
  const handle = file.open(FileMode.WriteOnly);
  let pending = '';
  let closed = false;
  const flush = () => {
    if (closed || !pending) return;
    handle.writeBytes(new TextEncoder().encode(pending));
    pending = '';
  };
  return {
    uri: file.uri,
    tag,
    append(text) {
      if (closed) throw new Error('Recording file is closed.');
      pending += text;
      if (pending.length >= 32768) flush();
    },
    flush,
    close() {
      if (closed) return;
      try { flush(); } finally { closed = true; handle.close(); }
    },
  };
}

/** Moves the camera's recorded clip (in cache) alongside its session's JSONL. */
export function saveRecordingVideo(tag, sourceUri) {
  const directory = recordingsDirectory();
  directory.create({ idempotent: true, intermediates: true });
  const source = new File(sourceUri);
  const dest = new File(directory, `imu-${tag}.mp4`);
  source.moveSync(dest);
  return { name: dest.name, uri: dest.uri, size: dest.size };
}

export function listRecordings() {
  const directory = recordingsDirectory();
  if (!directory.exists) return [];
  const files = directory.list();
  return files.filter((file) => file instanceof File && file.name.endsWith('.jsonl'))
    .sort((a, b) => b.name.localeCompare(a.name))
    .map((file) => {
      const tag = file.name.slice('imu-'.length, -'.jsonl'.length);
      const video = files.find((f) => f instanceof File && f.name === `imu-${tag}.mp4`);
      return {
        name: file.name, uri: file.uri, size: file.size,
        video: video ? { name: video.name, uri: video.uri, size: video.size } : null,
      };
    });
}
