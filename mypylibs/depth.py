"""Step 6: dense depth from each frame and a later frame of the same scene video.

The partner frame is picked by depth_pairs (mypylibs/pose.py) so the features
moved enough for useful disparity. t from recoverPose is a unit vector; its length
in cm is taken as cm_per_frame * gap, i.e. assuming the camera slides at a steady speed.
"""

import csv
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mypylibs.pose import depth_pairs, rotation_deg
from mypylibs.video import read_frames


# Pairs are picked at ~10 px median motion, so near objects can move several times that
NUM_DISPARITIES = 96
BLOCK = 7  # SGBM window; 5 left too many speckles on the low-texture desk and walls


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

    # A mostly vertical t gives vertical epipolar lines; transposing both images
    # turns them into rows, which is all StereoSGBM can search along.
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])

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

    # With equal rectified principal points, depth = -P2[0, 3] / disparity
    # (P2[1, 3] for vertical pairs).
    numerator = -P2[1, 3] if vertical else -P2[0, 3]
    minimum = 0 if numerator > 0 else 1 - NUM_DISPARITIES

    matcher = cv2.StereoSGBM_create(
        minDisparity=minimum,
        numDisparities=NUM_DISPARITIES,
        blockSize=BLOCK,
        P1=8 * BLOCK ** 2,
        P2=32 * BLOCK ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    gray = [
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for frame in rectified
    ]
    if vertical:
        gray = [g.T.copy() for g in gray]
        roi1, roi2 = [(y, x, h, w) for x, y, w, h in (roi1, roi2)]
    disparity = matcher.compute(*gray).astype(np.float32) / 16.0

    x, y, w, h = cv2.getValidDisparityROI(
        roi1, roi2, minimum, NUM_DISPARITIES, BLOCK
    )

    valid = np.zeros(disparity.shape, dtype=bool)
    valid[y:y + h, x:x + w] = True
    valid &= disparity >= minimum
    # Median filter removes isolated mismatches without blurring depth edges much
    disparity = cv2.medianBlur(disparity, 5)
    valid &= disparity * numerator > 0
    valid &= np.abs(disparity) >= 0.5

    if vertical:
        disparity, valid = disparity.T, valid.T

    if valid.mean() < 0.05:
        raise ValueError("Less than 5 percent usable depth")

    depth = np.full((height, width), np.nan, dtype=np.float32)
    np.divide(numerator, disparity, out=depth, where=valid)

    return rectified[0], depth



def make_depth_video(video, K, D, output, cm_per_frame):
    frames = read_frames(video)
    if len(frames) < 2:
        raise ValueError("Not enough frames for a stereo pair")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    poses = depth_pairs(frames, K, D)
    records = []
    results = {}

    for i, pose in enumerate(poses):
        j = pose["partner"] if pose is not None else ""
        status = "Pose unavailable"
        coverage = 0.0

        if pose is not None:
            try:
                t = pose["t"]
                t_cm = t * (cm_per_frame * (j - i) / np.linalg.norm(t))

                image, depth = compute_depth(
                    frames[i], frames[j],
                    K, D, pose["R"], t_cm,
                )
                results[i] = (image, depth, j)
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
        for _, depth, _ in results.values()
    ])
    low, high = np.percentile(samples, [5, 95])
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
        f"Stereo depth | assumed camera speed {cm_per_frame:g} cm/frame"
    )
    for ax in axes:
        ax.axis("off")

    writer = None
    first_valid = min(results)

    try:
        # Frames without a usable pair keep showing the last one, labelled with its frames,
        # so the video doesn't flicker to blank on every rejected pair
        shown = None
        for i in range(len(frames)):
            if i in results:
                shown = i

            if shown is None:
                scene_plot.set_data(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
                depth_plot.set_data(blank)
                axes[0].set_title(f"Frame {i}")
                axes[1].set_title("Depth unavailable")
            else:
                image, depth, j = results[shown]
                scene_plot.set_data(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                depth_plot.set_data(depth)
                axes[0].set_title(f"Frame {shown}, rectified")
                held = "" if shown == i else f", held at frame {i}"
                axes[1].set_title(
                    f"Frames {shown} to {j} (baseline {cm_per_frame * (j - shown):.1f} cm){held}"
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

            if i == first_valid:
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