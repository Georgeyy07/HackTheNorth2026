"""
Pothole detection on video using the locally trained YOLO26 weights.

Runs every (or every Nth) frame of a video through the trained model,
draws detections on it, writes an annotated output video, and prints a
console log of when potholes showed up and how confident the model was.

Usage:
    python detect_potholes.py --video input.mp4 --weights weights/best.pt

Requires:
    pip install ultralytics opencv-python-headless
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from tiger_client import DEFAULT_TIGER_URL, report_detection


def run(video_path, weights_path, output_path, conf, crack_conf, vid_stride,
        tiger_url=None, lat=None, lon=None):
    if not Path(weights_path).exists():
        raise SystemExit(f"Weights not found: {weights_path}")
    if not Path(video_path).exists():
        raise SystemExit(f"Video not found: {video_path}")

    model = YOLO(weights_path)
    crack_cls_id = next(
        (i for i, name in model.names.items() if name.lower() == "crack"), None
    )
    # floor conf passed to the model itself so ultralytics's own NMS doesn't
    # discard low-confidence cracks before we get a chance to per-class filter
    predict_conf = min(conf, crack_conf) if crack_cls_id is not None else conf

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps / vid_stride, (width, height))

    print(f"Video: {video_path} | fps={fps:.1f} | weights={weights_path} | "
          f"conf>={conf} (crack>={crack_conf}) | vid_stride={vid_stride}")

    total_detections = 0
    frame_idx = 0

    for result in model.predict(
        source=video_path, stream=True, conf=predict_conf, vid_stride=vid_stride,
        verbose=False,
    ):
        if crack_cls_id is not None:
            cls = result.boxes.cls
            confs = result.boxes.conf
            keep = (cls == crack_cls_id) & (confs >= crack_conf) | \
                   (cls != crack_cls_id) & (confs >= conf)
            result = result[keep]

        writer.write(result.plot())

        n = len(result.boxes)
        if n:
            timestamp = (frame_idx * vid_stride) / fps
            total_detections += n
            labels = [
                f"{result.names[int(c)]}:{conf:.2f}"
                for c, conf in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist())
            ]
            print(f"  [{timestamp:6.2f}s] {n} detection(s): {labels}")

            if tiger_url and lat is not None and lon is not None:
                for box_conf in result.boxes.conf.tolist():
                    report_detection(lat, lon, box_conf, tiger_url=tiger_url)

        frame_idx += 1

    writer.release()
    print(f"\nDone. {total_detections} total detections across "
          f"{frame_idx} processed frames.")
    print(f"Annotated video saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--weights", default="weights/best.pt",
                         help="Path to trained .pt weights (default: weights/best.pt)")
    parser.add_argument("--output", default="annotated_output.mp4",
                         help="Path to save the annotated output video")
    parser.add_argument("--conf", type=float, default=0.4,
                         help="Minimum confidence to keep a pothole detection (default 0.4)")
    parser.add_argument("--crack-conf", type=float, default=0.2,
                         help="Minimum confidence to keep a crack detection (default 0.2, "
                              "lower than --conf so more crack guesses show up)")
    parser.add_argument("--vid-stride", type=int, default=1,
                         help="Only run inference every Nth frame (default 1 = every frame)")
    parser.add_argument("--tiger-url", nargs="?", const=DEFAULT_TIGER_URL, default=None,
                         help=f"Report every detection to TigerDB via the road_viewer API "
                              f"(POST /api/potholes). Defaults to {DEFAULT_TIGER_URL} if given "
                              f"with no value. Requires --lat/--lon.")
    parser.add_argument("--lat", type=float, help="Latitude to attach to reported detections")
    parser.add_argument("--lon", type=float, help="Longitude to attach to reported detections")
    args = parser.parse_args()
    if args.tiger_url and (args.lat is None or args.lon is None):
        parser.error("--tiger-url requires --lat and --lon")

    run(args.video, args.weights, args.output, args.conf, args.crack_conf, args.vid_stride,
        tiger_url=args.tiger_url, lat=args.lat, lon=args.lon)
