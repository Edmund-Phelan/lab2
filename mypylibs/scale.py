"""Camera speed (cm/frame) per scene, i.e. the scale of the depth step. Run after main.py:

    python -m mypylibs.scale

recoverPose's t has no length, so the depth step assumes the phone slides at a steady
speed (runner.CM_PER_FRAME). The phones' video metadata has no subject distance or
motion data, so the speed comes from something of known size in each scene:

  01_01  the calibration board lies on the desk: solvePnP gives the camera position in
         every frame where it is detected, and the speed is measured directly.
  02_01  a mug, assumed 9.5 cm tall (152 px at frame 67, so ~39 cm away; its 8.2 cm
         width gives 38.5 cm).
  03_01  the building wall, ~18 m away at frame 129: storeys repeat every ~95 px (3 m
         floor to floor -> 17.8 m) and the doors are ~60 px tall (2 m -> 18.8 m).

For 02 and 03 the speed is the one that makes the median depth over the object match
that distance, taken as the median over the depth pairs near the reference frame. The
pairs disagree by up to 3x, partly because the hand doesn't move at a steady speed.
"""

import csv
from os import path as opath

import cv2
import numpy as np

from mypylibs.calibration import board_points
from mypylibs.depth import compute_depth
from mypylibs.frames import PATTERN
from mypylibs.pose import detect, estimate_pose, match_features
from mypylibs.runner import ROOT_IN, ROOT_OUT
from mypylibs.video import read_frames

# scene: (reference frame, object region x0, x1, y0, y1 in px, distance in cm)
OBJECTS = {
    '02_01': (67, (180, 280, 175, 290), 39.0),     # mug
    '03_01': (129, (20, 360, 130, 260), 1800.0),   # building wall
}
NEAR = 8  # use depth pairs starting within this many frames of the reference frame


def load(scene: str) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    cam = scene[:2]
    K = np.loadtxt(opath.join(ROOT_OUT, f'{cam}_K.csv'), delimiter=',')
    D = np.loadtxt(opath.join(ROOT_OUT, f'{cam}_D.csv'), delimiter=',')
    return read_frames(opath.join(ROOT_IN, f'{scene}.mp4')), K, D


def board_speed(scene: str) -> float:
    """Camera speed from the board's pose in every frame where the whole board is found.

    The board is small and seen at a grazing angle, so frames are upscaled 2x and the
    classic detector is used: the sector-based one finds nothing in 01_01.
    """
    frames, K, D = load(scene)
    obj, centres = board_points(), {}
    for i, frame in enumerate(frames):
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)
        found, corners = cv2.findChessboardCorners(
            gray, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not found:
            continue
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)) / 2
        _, rvec, tvec = cv2.solvePnP(obj, corners, K, D)
        centres[i] = (-cv2.Rodrigues(rvec)[0].T @ tvec).ravel()
    idx = sorted(centres)
    if len(idx) < 2:
        raise RuntimeError(f'{scene}: board found in {len(idx)} frames, need 2')
    distance = np.linalg.norm(centres[idx[-1]] - centres[idx[0]])
    print(f'{scene}: board in frames {idx}, camera moved {distance:.2f} cm')
    return float(distance / (idx[-1] - idx[0]))


def object_speed(scene: str) -> float:
    """Speed that puts the object at its known distance (median over nearby depth pairs)."""
    ref, (x0, x1, y0, y1), distance = OBJECTS[scene]
    frames, K, D = load(scene)
    with open(opath.join(ROOT_OUT, scene.replace('_', '-') + '_status.csv')) as f:
        pairs = {int(r['first_frame']): int(r['second_frame'])
                 for r in csv.DictReader(f) if r['status'] == 'OK'}
    xs, ys = np.meshgrid(np.arange(x0, x1, 4), np.arange(y0, y1, 4))
    pts = np.float32(np.c_[xs.ravel(), ys.ravel()]).reshape(-1, 1, 2)
    h, w = frames[0].shape[:2]

    speeds = []
    for i in sorted(p for p in pairs if abs(p - ref) <= NEAR):
        j = pairs[i]
        pose = estimate_pose(*match_features(detect(frames[i]), detect(frames[j])), K, D)
        t = pose['t'] * (j - i) / np.linalg.norm(pose['t'])  # depth at 1 cm/frame
        _, depth = compute_depth(frames[i], frames[j], K, D, pose['R'], t)
        # The depth map is in rectified coordinates: move the object's pixels there too
        R1, _, P1, _, _, _, _ = cv2.stereoRectify(K, D, K, D, (w, h), pose['R'], t,
                                                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        r = np.round(cv2.undistortPoints(pts, K, D, R=R1, P=P1).reshape(-1, 2)).astype(int)
        r = r[(r[:, 0] >= 0) & (r[:, 0] < w) & (r[:, 1] >= 0) & (r[:, 1] < h)]
        z = depth[r[:, 1], r[:, 0]]
        z = z[np.isfinite(z)]
        if len(z):
            speeds.append(distance / np.median(z))
            print(f'{scene} frames {i}->{j}: {speeds[-1]:.3f} cm/frame')
    return float(np.median(speeds))


if __name__ == '__main__':
    print(f"01_01: {board_speed('01_01'):.3f} cm/frame")
    for scene in OBJECTS:
        print(f'{scene}: {object_speed(scene):.3f} cm/frame')
