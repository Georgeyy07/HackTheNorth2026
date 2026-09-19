import { Directory, File, FileMode, Paths } from 'expo-file-system';

const recordingsDirectory = () => new Directory(Paths.document, 'motion-recordings');

export function createRecordingSink() {
  const directory = recordingsDirectory();
  directory.create({ idempotent: true, intermediates: true });
  const name = `imu-${new Date().toISOString().replace(/[:.]/g, '-')}-${Math.random().toString(36).slice(2, 8)}.jsonl`;
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

export function listRecordings() {
  const directory = recordingsDirectory();
  if (!directory.exists) return [];
  return directory.list().filter((file) => file instanceof File && file.name.endsWith('.jsonl'))
    .sort((a, b) => b.name.localeCompare(a.name))
    .map((file) => ({ name: file.name, uri: file.uri, size: file.size }));
}
