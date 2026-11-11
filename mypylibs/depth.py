"""Step 6 using the existing SIFT poses and 10-frame spacing."""

import csv
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mypylibs.pose import rotation_deg, scene_poses
from mypylibs.video import read_frames


def compute_depth(first, second, K, D, R, t):
    height, width = first.shape[:2]
    size = (width, height)

    if rotation_deg(R) > 10 or np.linalg.norm(t) < 1e-8:
        raise ValueError("Rotation too large or zero baseline")

    R1, R2, P1, P2, _, roi1, roi2 = cv2.stereoRectify(
        K, D, K, D, size, R, t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    if abs(P2[1, 3]) >= abs(P2[0, 3]):
        raise ValueError("Vertical stereo geometry")

    if not np.isfinite(P1).all() or not 0.5 < P1[0, 0] / K[0, 0] < 2:
        raise ValueError("Extreme rectification zoom")

    rectified = []
    for frame, rotation, projection in [
        (first, R1, P1),
        (second, R2, P2),
    ]:
        maps = cv2.initUndistortRectifyMap(
            K, D, rotation, projection, size, cv2.CV_32FC1
        )
        rectified.append(
            cv2.remap(frame, *maps, cv2.INTER_LINEAR)
        )

    # With equal rectified principal points, depth = -P2[0, 3] / disparity.
    numerator = -P2[0, 3]
    minimum = 0 if numerator > 0 else -63

    matcher = cv2.StereoSGBM_create(
        minDisparity=minimum,
        numDisparities=64,
        blockSize=5,
        P1=200,
        P2=800,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=50,
        speckleRange=2,
    )

    gray = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for frame in rectified
    ]
    disparity = matcher.compute(*gray).astype(np.float32) / 16.0

    x, y, w, h = cv2.getValidDisparityROI(
        roi1, roi2, minimum, 64, 5
    )

    valid = np.zeros((height, width), dtype=bool)
    valid[y:y + h, x:x + w] = True
    valid &= disparity >= minimum
    valid &= disparity * numerator > 0
    valid &= np.abs(disparity) >= 0.5

    if valid.mean() < 0.05:
        raise ValueError("Less than 5 percent usable depth")

    depth = np.full((height, width), np.nan, dtype=np.float32)
    np.divide(numerator, disparity, out=depth, where=valid)

    return rectified[0], depth



def make_depth_video(video, K, D, output, step=10, baseline_cm=None):
    frames = read_frames(video)
    if step < 1 or len(frames) <= step:
        raise ValueError("Not enough frames for the requested gap")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    poses = scene_poses(frames, K, D, step=step)
    records = []
    results = {}

    for i, pose in enumerate(poses):
        j = i + step
        status = "Pose unavailable"
        coverage = 0.0

        if pose is not None:
            try:
                t = pose["t"]
                t_cm = t * (baseline_cm / np.linalg.norm(t))

                image, depth = compute_depth(
                    frames[i], frames[j],
                    K, D, pose["R"], t_cm,
                )
                results[j] = (image, depth)
                coverage = float(np.isfinite(depth).mean())
                status = "OK"
            except ValueError as exc:
                status = str(exc)

        records.append((i, j, status, coverage))
        print(f"Frames {i} -> {j}: {status}", flush=True)

    output_base = os.path.splitext(output)[0]
    with open(output_base + "_status.csv", "w", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            ["first_frame", "second_frame", "status", "valid_fraction"]
        )
        csv_writer.writerows(records)

    if not results:
        raise RuntimeError("No usable depth pairs. See the status CSV.")

    # Keep the same display scale throughout the video.
    samples = np.concatenate([
        depth[np.isfinite(depth)][::20]
        for _, depth in results.values()
    ])
    low, high = np.percentile(samples, [2, 98])
    high = max(high, low + 1e-6)

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4), dpi=100,
        constrained_layout=True,
    )
    cmap = plt.get_cmap("turbo_r").copy()
    cmap.set_bad("#303030")

    blank = np.full(frames[0].shape[:2], np.nan)
    scene_plot = axes[0].imshow(np.zeros_like(frames[0]))
    depth_plot = axes[1].imshow(
        blank, cmap=cmap, vmin=low, vmax=high
    )

    fig.colorbar(
        depth_plot,
        ax=axes[1],
        label="Estimated depth (cm)",
    )
    fig.suptitle(
        f"Stereo depth | assumed baseline {baseline_cm:g} cm across {step} frames"
    )
    for ax in axes:
        ax.axis("off")

    writer = None
    first_valid = min(results)

    try:
        for j in range(step, len(frames)):
            image, depth = results.get(
                j, (frames[j - step], blank)
            )

            scene_plot.set_data(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )
            depth_plot.set_data(depth)

            axes[0].set_title(
                "Rectified first frame"
                if j in results else "Original first frame"
            )
            axes[1].set_title(
                f"Frames {j - step} to {j}"
                if j in results else "Depth unavailable"
            )

            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            video_frame = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

            if writer is None:
                height, width = video_frame.shape[:2]
                writer = cv2.VideoWriter(
                    output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not open video writer")

            writer.write(video_frame)

            if j == first_valid:
                fig.savefig(
                    output_base + "_example.png", dpi=150
                )
    finally:
        if writer is not None:
            writer.release()
        plt.close(fig)

    print(
        f"Saved {output}: "
        f"{len(results)}/{len(poses)} usable pairs"
    )