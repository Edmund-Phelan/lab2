"""Step 2: detect checkerboard inner corners. Step 3: camera intrinsics."""

from glob import glob

import cv2
import numpy as np

from mypylibs.frames import PATTERN

SQUARE_CM = 2.5  # checkerboard square size; sets the unit (cm) of all 3D results


def find_corners(frame: np.ndarray, pattern: tuple[int, int] = PATTERN) -> np.ndarray | None:
    """Inner corners of the board as (N, 1, 2) float32 sub-pixel coords, or None if not found.

    Uses the sector-based detector (findChessboardCornersSB), which is sub-pixel
    accurate on its own. findChessboardCorners + cornerSubPix put two corners in
    the wrong place on washed-out frame 13 (1.2 px error vs 0.12 px with SB).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags)
    return corners.reshape(-1, 1, 2).astype(np.float32) if found else None


def find_all_corners(frames: list[tuple[int, np.ndarray]], pattern: tuple[int, int] = PATTERN
                     ) -> tuple[list[np.ndarray], list[int], tuple[int, int]]:
    """Corners and frame index for every frame where the board is found, plus the image size (w, h)."""
    image_points, used = [], []
    for idx, frame in frames:
        corners = find_corners(frame, pattern)
        if corners is None:
            print(f'frame {idx}: board NOT found by the accurate detector, skipped')
            continue
        image_points.append(corners)
        used.append(idx)
    print(f'Corners found in {len(image_points)}/{len(frames)} frames')
    return image_points, used, frames[0][1].shape[1::-1]


def draw_corners(frame: np.ndarray, corners: np.ndarray, pattern: tuple[int, int] = PATTERN) -> np.ndarray:
    """Copy of frame with the detected corners drawn (for checking / the report)."""
    out = frame.copy()
    cv2.drawChessboardCorners(out, pattern, corners, True)
    return out


def board_points(pattern: tuple[int, int] = PATTERN, s: float = SQUARE_CM) -> np.ndarray:
    """3D corners (i*s, j*s, 0) on the Z=0 plane, in the order OpenCV detects them."""
    cols, rows = pattern
    pts = np.zeros((cols * rows, 3), np.float32)
    pts[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * s
    return pts


def calibrate(image_points: list[np.ndarray], size: tuple[int, int],
              pattern: tuple[int, int] = PATTERN, s: float = SQUARE_CM) -> dict:
    """Run cv2.calibrateCamera on the detected corners.

    Returns K (3x3), D (k1, k2, p1, p2, k3), OpenCV's RMS error, and the mean
    reprojection error (px) overall and per image.
    """
    object_points = [board_points(pattern, s)] * len(image_points)
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, size, None, None)

    # Mean Euclidean distance between detected and reprojected corners
    per_image = []
    for o, ip, r, t in zip(object_points, image_points, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, D)
        per_image.append(float(np.linalg.norm(proj.reshape(-1, 2) - ip.reshape(-1, 2), axis=1).mean()))

    return {'K': K, 'D': D.ravel()[:5], 'rms': rms,
            'mean_error': float(np.mean(per_image)), 'per_image_error': per_image}


if __name__ == '__main__':
    # Self-check on every in/XX_calibration.mp4: all 8x5 corners found inside the image, error <= 0.5 px
    import sys
    from mypylibs.frames import extract_calibration_frames
    for video in sys.argv[1:] or sorted(glob('in/[0-9][0-9]_calibration.mp4')):
        frames = extract_calibration_frames(video)
        pts, _, (w, h) = find_all_corners(frames)
        assert len(pts) >= 15, f'{video}: board found in too few frames'
        for c in pts:
            assert c.shape == (PATTERN[0] * PATTERN[1], 1, 2)
            assert (c[..., 0] >= 0).all() and (c[..., 0] < w).all()
            assert (c[..., 1] >= 0).all() and (c[..., 1] < h).all()
        err = calibrate(pts, (w, h))['mean_error']
        print(f'{video}: mean reprojection error {err:.3f} px')
        assert err <= 0.5, f'{video}: reprojection error over 0.5 px'
    print('OK')
