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


def run(video_path, weights_path, output_path, conf, vid_stride):
    if not Path(weights_path).exists():
        raise SystemExit(f"Weights not found: {weights_path}")
    if not Path(video_path).exists():
        raise SystemExit(f"Video not found: {video_path}")

    model = YOLO(weights_path)

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
          f"conf>={conf} | vid_stride={vid_stride}")

    total_detections = 0
    frame_idx = 0

    for result in model.predict(
        source=video_path, stream=True, conf=conf, vid_stride=vid_stride,
        verbose=False,
    ):
        writer.write(result.plot())

        n = len(result.boxes)
        if n:
            timestamp = (frame_idx * vid_stride) / fps
            total_detections += n
            confs = [f"{c:.2f}" for c in result.boxes.conf.tolist()]
            print(f"  [{timestamp:6.2f}s] {n} pothole(s) detected, "
                  f"confidence={confs}")

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
                         help="Minimum confidence to keep a detection (default 0.4)")
    parser.add_argument("--vid-stride", type=int, default=1,
                         help="Only run inference every Nth frame (default 1 = every frame)")
    args = parser.parse_args()

    run(args.video, args.weights, args.output, args.conf, args.vid_stride)
