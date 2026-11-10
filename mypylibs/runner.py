"""Per-camera and per-scene pipeline steps, called by main.py.

A camera XX is any in/XX_calibration.mp4; its scenes are in/XX_YY.mp4. Adding a
camera = dropping its two (pre-processed, see mypylibs/video.py) videos into in/.

    01  Samsung Galaxy Z Fold6 (SM-F956B)
    02  iPhone 15
    03  iPhone 15 Pro
"""

import re
from glob import glob
from os import path as opath

import numpy as np

from mypylibs.calibration import calibrate, find_all_corners
from mypylibs.frames import extract_calibration_frames
from mypylibs.visualize import make_video as calibration_video
from mypylibs.visualize_pose import make_outputs as pose_outputs
from mypylibs.visualize_scene import make_video as motion_video

ROOT_IN = 'in'
ROOT_OUT = 'out'


def cameras() -> list[str]:
    """Camera ids ('01', '02', ...) that have a calibration video."""
    paths = glob(opath.join(ROOT_IN, '[0-9][0-9]_calibration.mp4'))
    return sorted(opath.basename(p)[:2] for p in paths)


def scenes(cam: str) -> list[str]:
    """Scene videos in/XX_YY.mp4 of camera `cam`."""
    paths = glob(opath.join(ROOT_IN, f'{cam}_*.mp4'))
    return sorted(p for p in paths if re.fullmatch(rf'{cam}_\d\d\.mp4', opath.basename(p)))


def run_calibration(cam: str) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate camera `cam` from 20 frames of in/XX_calibration.mp4; save K, D to out/."""
    video = opath.join(ROOT_IN, f'{cam}_calibration.mp4')
    image_points, used, size = find_all_corners(extract_calibration_frames(video))

    res = calibrate(image_points, size)
    K, D = res['K'], res['D']

    print(f'Camera {cam}: {len(image_points)} images, {size[0]}x{size[1]}')
    print(f'  fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f} skew={K[0, 1]:.2f}')
    print(f'  k1={D[0]:.4f} k2={D[1]:.4f} p1={D[2]:.4f} p2={D[3]:.4f} k3={D[4]:.4f}')
    print(f'  reprojection error: mean={res["mean_error"]:.3f} px, rms={res["rms"]:.3f} px')
    for idx, e in zip(used, res['per_image_error']):
        print(f'    frame {idx}: {e:.3f} px')

    np.savetxt(opath.join(ROOT_OUT, f'{cam}_K.csv'), K, delimiter=',', fmt='%.6f')
    np.savetxt(opath.join(ROOT_OUT, f'{cam}_D.csv'), D[None], delimiter=',', fmt='%.6f')
    calibration_video(video, K, D, used, opath.join(ROOT_OUT, f'{cam}_calibration_corners.mp4'))
    return K, D


def run_scene(video: str, K: np.ndarray, D: np.ndarray) -> None:
    """Motion, pose and trajectory outputs for one scene video."""
    name = opath.splitext(opath.basename(video))[0]
    print(f'--- Scene {name} ---')
    motion_video(video, opath.join(ROOT_OUT, f'{name}_motion.mp4'))
    pose_outputs(video, K, D, opath.join(ROOT_OUT, name))
