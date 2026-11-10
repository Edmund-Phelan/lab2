"""Pick sharp calibration frames that show the full checkerboard."""

import cv2
import numpy as np

# Inner corners (cols, rows). The printed board has 9x6 squares = 8x5 inner corners
# (the README's "9x6 inner grid" is actually the square count).
PATTERN = (8, 5)


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: higher means sharper (less blur)."""
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_calibration_frames(video: str, n: int = 20) -> list[tuple[int, np.ndarray]]:
    """(frame index, frame) for n frames of the video, in time order.

    The video is split into n equal time windows and, from each, the sharpest
    frame where the whole checkerboard is detected is kept. Spreading the picks
    over time gives a variety of board poses.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {video}')

    # (frame index, sharpness, frame) for every frame with a detected board
    candidates = []
    total = 0
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCorners(gray, PATTERN, flags)
        if found:
            candidates.append((total, sharpness(gray), frame))
        total += 1
    cap.release()

    print(f'Board found in {len(candidates)}/{total} frames')
    if len(candidates) < n:
        raise RuntimeError(f'Only {len(candidates)} frames show the board, need {n}')

    # Sharpest frame in each time window
    picks = []
    for i in range(n):
        lo, hi = i * total / n, (i + 1) * total / n
        window = [c for c in candidates if lo <= c[0] < hi]
        if window:
            picks.append(max(window, key=lambda c: c[1]))

    # Windows without a detected board: fill with the frames farthest in time
    # from the current picks (likely the most different poses), not near-duplicates
    missing = n - len(picks)
    for _ in range(missing):
        used = {p[0] for p in picks}
        rest = [c for c in candidates if c[0] not in used]
        picks.append(max(rest, key=lambda c: (min(abs(c[0] - p[0]) for p in picks), c[1])))
    if missing:
        print(f'{missing} time windows had no board; filled with frames far from other picks')

    picks.sort(key=lambda c: c[0])
    print('Calibration frames: ' + ', '.join(str(idx) for idx, _, _ in picks))
    return [(idx, frame) for idx, _, frame in picks]
