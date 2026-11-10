"""Step 2: detect checkerboard inner corners. Step 3: camera intrinsics."""

from glob import glob
from os import path as opath

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


def load_calibration_images(in_dir: str) -> list[str]:
    """Paths of in/1.png ... in/N.png, in numeric order."""
    paths = glob(opath.join(in_dir, '[0-9]*.png'))
    return sorted(paths, key=lambda p: int(opath.splitext(opath.basename(p))[0]))


def find_all_corners(paths: list[str], pattern: tuple[int, int] = PATTERN
                     ) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Corners for every image where the board is found, plus the image size (w, h)."""
    image_points, size = [], None
    for p in paths:
        frame = cv2.imread(p)
        size = frame.shape[1::-1]
        corners = find_corners(frame, pattern)
        if corners is None:
            print(f'{p}: board NOT found, skipped')
            continue
        image_points.append(corners)
    print(f'Corners found in {len(image_points)}/{len(paths)} images')
    return image_points, size


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
    # Self-check: every calibration image yields all 8x5 corners inside the image
    paths = load_calibration_images('in')
    assert paths, 'no calibration images in in/'
    pts, (w, h) = find_all_corners(paths)
    assert len(pts) == len(paths), 'board missed in some images'
    for c in pts:
        assert c.shape == (PATTERN[0] * PATTERN[1], 1, 2)
        assert (c[..., 0] >= 0).all() and (c[..., 0] < w).all()
        assert (c[..., 1] >= 0).all() and (c[..., 1] < h).all()
    preview = 'out/corners_preview.png'
    cv2.imwrite(preview, draw_corners(cv2.imread(paths[0]), pts[0]))
    print(f'OK, preview written to {preview}')
