"""Step 5 visualisation: per-pair pose overlay video and the chained camera trajectory.

Called from main.py; writes out/XX_YY_pose.mp4 and out/XX_YY_trajectory.png. Pairs are
the ones the depth step uses (depth_pairs: frame t and the first frame that moved enough).
The video shows the partner frame (undistorted) with RANSAC inlier matches in green,
outliers in red, and a few epipolar lines: if the pose is right, each inlier sits on its line.
"""

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mypylibs.pose import depth_pairs, rotation_deg
from mypylibs.video import read_frames
from mypylibs.visualize import RED, label

GREEN = (0, 220, 0)
N_EPILINES = 8


def draw_pair(frame: np.ndarray, pose: dict | None, t: int) -> np.ndarray:
    """Undistorted partner frame with matches, epipolar lines and pose text."""
    out = frame.copy()
    if pose is None:
        label(out, f'frame {t}', (10, 22))
        label(out, 'pose REJECTED (too little parallax / ambiguous / points behind camera)', (10, 44), RED)
        return out
    label(out, f'frame {t} -> {pose["partner"]}', (10, 22))

    p1, p2, m = pose['pts1'], pose['pts2'], pose['mask']
    for (x0, y0), (x1, y1), ok in zip(p1, p2, m):
        c = GREEN if ok else RED
        cv2.line(out, (round(x0), round(y0)), (round(x1), round(y1)), c, 1, cv2.LINE_AA)
        cv2.circle(out, (round(x1), round(y1)), 2, c, -1, cv2.LINE_AA)

    # Epipolar line in frame 2 of a frame-1 inlier: l2 = F x1; its match should lie on it
    w = out.shape[1]
    for x in p1[m][:: max(1, m.sum() // N_EPILINES)][:N_EPILINES]:
        a, b, c = pose['F'] @ np.array([x[0], x[1], 1.0])
        if abs(b) > 1e-9:
            cv2.line(out, (0, round(-c / b)), (w, round(-(c + a * w) / b)), (255, 200, 0), 1, cv2.LINE_AA)

    tx, ty, tz = pose['t'].ravel()
    label(out, f'{pose["inliers"]}/{pose["matches"]} inliers   rot {rotation_deg(pose["R"]):.2f} deg', (10, 44))
    label(out, f't = ({tx:+.2f}, {ty:+.2f}, {tz:+.2f})  (unit length, scale unknown)', (10, 66))
    return out


def trajectory(poses: list[dict | None]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Camera centres, viewing directions and frame numbers, chaining pairs t -> partner -> ...

    recoverPose gives X2 = R X1 + t, so camera 2's centre in camera-1 coords is -R^T t
    and its axes are R^T. t has no scale, so each hop is scaled by its gap in frames
    (steady camera speed, as in the depth step). Where a frame has no pose, the chain
    restarts from the next frame that has one, as if the camera hadn't moved in between.
    """
    R_w, C_w = np.eye(3), np.zeros(3)
    centres, dirs, frames = [C_w], [R_w[:, 2]], [0]
    t = 0
    while t < len(poses):
        p = poses[t]
        if p is None:
            t += 1
            continue
        gap = p['partner'] - t
        C_w = C_w + R_w @ (-p['R'].T @ p['t'].ravel()) * gap
        R_w = R_w @ p['R'].T
        centres.append(C_w)
        dirs.append(R_w[:, 2])
        frames.append(p['partner'])
        t = p['partner']
    return np.array(centres), np.array(dirs), frames


def plot_trajectory(centres: np.ndarray, dirs: np.ndarray, frames: list[int], dst: str) -> None:
    """Top-down (X-Z) camera path with an arrow for where each camera looks."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(centres[:, 0], centres[:, 2], 'o-', ms=4)
    ax.quiver(centres[:, 0], centres[:, 2], dirs[:, 0], dirs[:, 2], angles='xy', scale=8, width=0.004, color='C1')
    for f, (x, _, z) in zip(frames, centres):
        ax.annotate(str(f), (x, z), textcoords='offset points', xytext=(3, 5), fontsize=7)
    ax.set_xlabel('X (frames of motion at steady speed)')
    ax.set_ylabel('Z (forward)')
    ax.set_title('Camera path, labelled by frame (top view)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def make_outputs(video: str, K: np.ndarray, D: np.ndarray, out_prefix: str) -> None:
    """Write <out_prefix>_pose.mp4 and <out_prefix>_trajectory.png for one scene."""
    frames = read_frames(video)
    poses = depth_pairs(frames, K, D)

    dst = f'{out_prefix}_pose.mp4'
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    for t, pose in enumerate(poses):
        # Points were undistorted, so draw on the undistorted frame to line up
        shown = frames[pose['partner'] if pose is not None else t]
        writer.write(draw_pair(cv2.undistort(shown, K, D), pose, t))
    writer.release()

    centres, dirs, hops = trajectory(poses)
    plot_trajectory(centres, dirs, hops, f'{out_prefix}_trajectory.png')
    kept = sum(p is not None for p in poses)
    print(f'{kept}/{len(poses)} poses kept; wrote {dst} and {out_prefix}_trajectory.png')
