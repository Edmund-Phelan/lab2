"""Step 2: detect checkerboard inner corners with sub-pixel precision."""

from glob import glob
from os import path as opath

import cv2
import numpy as np

from mypylibs.frames import PATTERN

# Stop sub-pixel refinement after 30 iterations or when the corner moves < 0.001 px
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(frame: np.ndarray, pattern: tuple[int, int] = PATTERN) -> np.ndarray | None:
    """Inner corners of the board as (N, 1, 2) float32 sub-pixel coords, or None if not found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, pattern, flags)
    if not found:
        return None
    # ponytail: fixed 11x11 window suits 640x360 frames; scale it if resolution changes
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)


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
