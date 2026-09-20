"""Build small, seek-friendly inspector videos without changing source timing.

Run after rendering/exporting recordings:
    python -m scripts.build_video_previews --export /path/to/replay
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile

import imageio_ffmpeg


def build_preview(source: Path):
    target = source.with_name("annotated.preview.mp4")
    if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return target
    # Publish atomically: viewers keep using the original until encoding ends.
    with tempfile.TemporaryDirectory(prefix="preview-", dir=source.parent) as temp:
        output = Path(temp) / "preview.mp4"
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-threads", "2", "-i", str(source), "-map", "0:v:0", "-an",
            "-vf", "scale=768:768:force_original_aspect_ratio=decrease:force_divisible_by=2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-threads", "2",
            "-maxrate", "900k", "-bufsize", "1800k", "-pix_fmt", "yuv420p",
            "-force_key_frames", "expr:gte(t,n_forced)", "-fps_mode", "passthrough",
            "-movflags", "+faststart", str(output),
        ], check=True)
        output.replace(target)
    print(f"{source.parent.name}: {source.stat().st_size:,} -> {target.stat().st_size:,} bytes", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    manifest = json.loads((args.export / "manifest.json").read_text())
    sources = [args.export / session["session_id"] / "annotated.mp4" for session in manifest["sessions"]]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(build_preview, (source for source in sources if source.is_file())))


if __name__ == "__main__":
    main()
