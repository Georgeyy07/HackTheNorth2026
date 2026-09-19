"""Plot actual Kaggle windows, all seven inputs, and the five-class targets.

Run: python plot_samples.py --split train
Requires Matplotlib in addition to the dataset's NumPy/PyTorch dependencies.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

if __package__:
    from road_training.dataset import CHANNELS, KAGGLE_CLASSES, RoadDataset
else:
    from road_training.dataset import CHANNELS, KAGGLE_CLASSES, RoadDataset


COLORS = ("#cbd5e1", "#0072b2", "#e69f00", "#cc79a7", "#d55e00")
AXIS_COLORS = ("#2563eb", "#9333ea", "#147d64")


def select_examples(dataset):
    """Select by labels, not signal strength; prefer complete isolated events."""
    pools = [[[], [], []] for _ in KAGGLE_CLASSES]
    for index in range(len(dataset)):
        item = dataset[index]
        y = item["labels"]["kaggle_type"].numpy()
        if (y == 0).all():
            pools[0][1].append(index)
            speed = item["x"][:, 6][item["mask"][:, 6]].numpy()
            # A moving car is a more useful normal-road example than parking.
            if speed.size and np.median(speed) >= 2:
                pools[0][0].append(index)
        for cls in range(1, len(KAGGLE_CLASSES)):
            if (y == cls).any() and (y == 0).any():
                pools[cls][2].append(index)
                if np.isin(y, [0, cls]).all():
                    pools[cls][1].append(index)
                    if y[0] == y[-1] == 0:
                        pools[cls][0].append(index)
    selected = []
    for cls, choices in enumerate(pools):
        candidates = next((indices for indices in choices if indices), [])
        if not candidates:
            raise ValueError(f"No example with normal context for {KAGGLE_CLASSES[cls]} in {dataset.split}")
        selected.append(candidates[len(candidates) // 2])
    return selected


def draw_figure(values, observed, labels, selection, split, normalized, output):
    """Every column shares an exact time axis across its traces and labels."""
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 5, figsize=(21, 9), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [2, 2, 1.3, .35]})
    t = np.arange(values.shape[1]) / 100
    duration = values.shape[1] / 100
    plotted = np.where(observed, values, np.nan)
    groups = (range(3), range(3, 6), range(6, 7))
    units = ("Acceleration (m/s²)", "Gyroscope (rad/s)", "Speed (m/s)")
    if normalized:
        units = ("Acceleration (z-score)", "Gyroscope (z-score)", "Speed (z-score)")
    for col, example in enumerate(selection):
        y = labels[col]
        edges = np.r_[0, np.flatnonzero(y[1:] != y[:-1]) + 1, len(y)]
        for start, stop in zip(edges[:-1], edges[1:]):
            cls = int(y[start])
            color = COLORS[cls] if cls >= 0 else "#64748b"
            for row in range(4):
                axes[row, col].axvspan(start / 100, stop / 100, color=color,
                                       alpha=1 if row == 3 else .15, linewidth=0)
        for row, channels in enumerate(groups):
            ax = axes[row, col]
            for j, channel in enumerate(channels):
                label = ("x", "y", "z")[j] if row < 2 else "speed"
                ax.plot(t, plotted[col, :, channel], color=AXIS_COLORS[j], linewidth=.8, label=label)
            ax.grid(axis="y", alpha=.2)
            ax.set_xlim(0, duration)
            if col == 0:
                ax.set_ylabel(units[row])
                ax.legend(loc="upper left", ncol=len(channels), fontsize=8, framealpha=.9)
        axes[0, col].set_title(f"{KAGGLE_CLASSES[col].capitalize()} example\n"
                               f"window {example['dataset_index']} · drive t={example['time_start_s']:.2f}s",
                               fontsize=11, pad=12)
        axes[3, col].set_yticks([])
        axes[3, col].set_xlabel("Time within window (s)")
        axes[3, col].set_ylim(0, 1)
    axes[3, 0].set_ylabel("Label")
    scale = "Fixed TRAIN z-scores" if normalized else "Raw sensor inputs (including gravity)"
    fig.suptitle(f"Kaggle {split.upper()} · normal road + four defect types\n{scale}",
                 fontsize=17, fontweight="bold", y=.985)
    legend = [Patch(facecolor=color, label=f"{i}: {name}")
              for i, (name, color) in enumerate(zip(KAGGLE_CLASSES, COLORS))]
    if (labels == -100).any():
        legend.append(Patch(facecolor="#64748b", label="−100: ignored"))
    fig.legend(handles=legend, loc="lower center", ncol=len(legend), bbox_to_anchor=(.5, .045), frameon=False)
    fig.text(.5, .018, "Shading and bottom strips show target labels, not model predictions. "
             "Normal outside annotations is assumed. Normal example prefers driving; no IMU amplitude ranking.",
             ha="center", fontsize=10, color="#475569")
    fig.subplots_adjust(left=.055, right=.99, top=.835, bottom=.16, wspace=.12, hspace=.17)
    stem = "inputs_zscore" if normalized else "inputs_raw"
    fig.savefig(output / f"{stem}.png", dpi=150, facecolor="white")
    fig.savefig(output / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    dataset = RoadDataset(args.data_root, split=args.split, source="real",
                          window_size=args.window_size, return_labels=True)
    indices = select_examples(dataset)
    examples = [dataset[index] for index in indices]
    x, mask, times = [np.stack([item[key].numpy() for item in examples]) for key in ("x", "mask", "time")]
    y = np.stack([item["labels"]["kaggle_type"].numpy() for item in examples])
    mean = np.asarray(dataset.train_stats["mean"], np.float32)
    std = np.asarray(dataset.train_stats["std"], np.float32)
    z = np.where(mask, (x - mean) / std, 0)
    selection = [dict(focus_class=KAGGLE_CLASSES[i], dataset_index=index,
                      recording_id=item["recording_id"], start_sample=item["start"],
                      time_start_s=float(item["time"][0]),
                      class_counts={name: int((y[i] == cls).sum()) for cls, name in enumerate(KAGGLE_CLASSES)},
                      ignored_samples=int((y[i] == -100).sum()))
                 for i, (index, item) in enumerate(zip(indices, examples))]
    output = args.output or Path(__file__).parent / "plots" / f"kaggle_five_class_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "windows.npz", x=x, mask=mask, time=times,
                        x_zscore=z, kaggle_type=y, channels=np.array(CHANNELS),
                        classes=np.array(KAGGLE_CLASSES))
    metadata = dict(split=args.split, source="real", sample_rate_hz=100, window_size=args.window_size,
                    classes=KAGGLE_CLASSES, examples=selection, train_statistics=dataset.train_stats,
                    selection_policy="Middle chronological candidate per class; normal prefers median observed speed >=2 m/s. Defects prefer a complete isolated event with normal on both sides, then isolated with partial context, then mixed. No acceleration/gyro amplitude ranking.",
                    manifest_sha256=hashlib.sha256((dataset.root / "manifest.json").read_bytes()).hexdigest())
    (output / "selection.json").write_text(json.dumps(metadata, indent=2) + "\n")
    draw_figure(x, mask, y, selection, args.split, False, output)
    draw_figure(z, mask, y, selection, args.split, True, output)
    print(f"Saved raw/normalized figures and exact input windows: {output.resolve()}")


if __name__ == "__main__":
    main()
