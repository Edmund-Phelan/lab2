"""Calibration video overlay: detected corners in red, board axes from the calibrated K, D.

Called from main.py. The frames chosen for calibration are labelled and held on screen briefly.
"""

import cv2
import numpy as np

from mypylibs.calibration import SQUARE_CM, board_points, find_corners

RED = (0, 0, 255)
WHITE = (255, 255, 255)
HOLD_FRAMES = 12  # how long each chosen calibration frame stays on screen


def label(img: np.ndarray, text: str, org: tuple[int, int], color=WHITE, scale: float = 0.5) -> None:
    """Text on a dark box so it reads on any background."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def overlay(frame: np.ndarray, K: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, bool]:
    """Frame with corners and board axes drawn, and whether the board was found."""
    out = frame.copy()
    corners = find_corners(frame)
    if corners is None:
        return out, False

    # Board pose from the calibrated intrinsics; axes (X red, Y green, Z blue) are 2 squares long
    _, rvec, tvec = cv2.solvePnP(board_points(), corners, K, D)
    cv2.drawFrameAxes(out, K, D, rvec, tvec, 2 * SQUARE_CM, 2)

    for x, y in corners.reshape(-1, 2):
        cv2.circle(out, (round(x), round(y)), 3, RED, -1, cv2.LINE_AA)
    dist = float(np.linalg.norm(tvec))
    label(out, f'board {dist:.0f} cm from camera', (10, out.shape[0] - 12))
    return out, True


def make_video(video: str, K: np.ndarray, D: np.ndarray, chosen: list[int], dst: str) -> None:
    """Overlay video; `chosen` are the frame indices used for calibration (held on screen)."""
    chosen = {idx: n + 1 for n, idx in enumerate(chosen)}

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None

    idx = found_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, found = overlay(frame, K, D)
        found_count += found
        if writer is None:
            h, w = out.shape[:2]
            writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        label(out, f'frame {idx}', (10, 22))
        label(out, f'fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}',
              (10, 44), scale=0.45)
        if not found:
            label(out, 'board not detected', (10, out.shape[0] - 12), RED)

        if idx not in chosen:
            writer.write(out)
        else:
            cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), RED, 4)
            label(out, f'calibration image {chosen[idx]}', (out.shape[1] - 200, 22), RED)
            for _ in range(HOLD_FRAMES):
                writer.write(out)
        idx += 1

    cap.release()
    writer.release()
    print(f'Board found in {found_count}/{idx} frames, wrote {dst}')


