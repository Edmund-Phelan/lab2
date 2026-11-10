"""Step 5 visualisation: per-pair pose overlay video and the chained camera trajectory.

Usage (from the repo root, after main.py has written out/XX_K.csv and out/XX_D.csv):
    python -m mypylibs.visualize_pose <scene_video> [camera_id]

Writes out/XX_YY_pose.mp4 and out/XX_YY_trajectory.png. The video shows frame t+STEP
(undistorted) with RANSAC inlier matches in green, outliers in red, and a few
epipolar lines: if the pose is right, each inlier sits on its line.
"""

import sys
from os import path as opath

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mypylibs.pose import rotation_deg, scene_poses
from mypylibs.video import read_frames
from mypylibs.visualize import RED, label

GREEN = (0, 220, 0)
N_EPILINES = 8
STEP = 10


def draw_pair(frame: np.ndarray, pose: dict | None, t: int) -> np.ndarray:
    """Undistorted frame t+STEP with matches, epipolar lines and pose text."""
    out = frame.copy()
    label(out, f'frame {t} -> {t + STEP}', (10, 22))
    if pose is None:
        label(out, 'pose REJECTED (too few matches / points behind camera)', (10, 44), RED)
        return out

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


def trajectory(poses: list[dict | None]) -> tuple[np.ndarray, np.ndarray]:
    """Camera centres and viewing directions, chaining non-overlapping pairs 0->STEP->2*STEP...

    recoverPose gives X2 = R X1 + t, so camera 2's centre in camera-1 coords is -R^T t
    and its axes are R^T. Stops at the first rejected pair (the chain can't bridge a gap).
    """
    R_w, C_w = np.eye(3), np.zeros(3)
    centres, dirs = [C_w], [R_w[:, 2]]
    for t in range(0, len(poses), STEP):
        p = poses[t]
        if p is None:
            print(f'Trajectory stops at frame {t}: pose rejected')
            break
        C_w = C_w + R_w @ (-p['R'].T @ p['t'].ravel())
        R_w = R_w @ p['R'].T
        centres.append(C_w)
        dirs.append(R_w[:, 2])
    return np.array(centres), np.array(dirs)


def plot_trajectory(centres: np.ndarray, dirs: np.ndarray, dst: str) -> None:
    """Top-down (X-Z) camera path with an arrow for where each camera looks."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(centres[:, 0], centres[:, 2], 'o-', ms=4)
    ax.quiver(centres[:, 0], centres[:, 2], dirs[:, 0], dirs[:, 2], angles='xy', scale=8, width=0.004, color='C1')
    for i, (x, _, z) in enumerate(centres):
        ax.annotate(str(i * STEP), (x, z), textcoords='offset points', xytext=(3, 5), fontsize=7)
    ax.set_xlabel('X (units of one pair baseline)')
    ax.set_ylabel('Z (forward)')
    ax.set_title(f'Camera path, every {STEP} frames (top view)')
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def main(video: str, cam: int) -> None:
    K = np.loadtxt(f'out/{cam:02}_K.csv', delimiter=',')
    D = np.loadtxt(f'out/{cam:02}_D.csv', delimiter=',')
    frames = read_frames(video)
    poses = scene_poses(frames, K, D, STEP)
    name = opath.splitext(opath.basename(video))[0]

    dst = f'out/{name}_pose.mp4'
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (w, h))
    for t, pose in enumerate(poses):
        # Points were undistorted, so draw on the undistorted frame to line up
        writer.write(draw_pair(cv2.undistort(frames[t + STEP], K, D), pose, t))
    writer.release()

    centres, dirs = trajectory(poses)
    plot_trajectory(centres, dirs, f'out/{name}_trajectory.png')
    kept = sum(p is not None for p in poses)
    print(f'{kept}/{len(poses)} poses kept; wrote {dst} and out/{name}_trajectory.png')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1)
