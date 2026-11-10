"""Video pre-processing shared by calibration and scene videos.

Every video (calibration and depth scenes) must go through preprocess_video
with the same settings, so the intrinsics stay valid for the scene videos.
"""

from os import path as opath

import cv2

MAX_LONG = 640    # max size of the long side (px)
MAX_SHORT = 360   # max size of the short side (px)
MAX_BYTES = 5 * 1024 * 1024


def target_size(w: int, h: int) -> tuple[int, int]:
    """Fit (w, h) inside 640x360 (or 360x640 for portrait), keeping aspect ratio."""
    long_side, short_side = max(w, h), min(w, h)
    scale = min(MAX_LONG / long_side, MAX_SHORT / short_side, 1.0)
    # Even dimensions keep video codecs happy
    return int(w * scale) // 2 * 2, int(h * scale) // 2 * 2


def preprocess_video(src: str, dst: str) -> None:
    """Downscale src to fit 640x360 and write it to dst as mp4."""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {src}')
    # Ignore the phone's portrait/landscape flag and keep raw sensor orientation,
    # so every video has the same pixel layout as the calibration video and K applies
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'Video has no frames: {src}')
    h, w = frame.shape[:2]
    size = target_size(w, h)

    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    while ok:
        writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        ok, frame = cap.read()

    cap.release()
    writer.release()

    n_bytes = opath.getsize(dst)
    print(f'{src} ({w}x{h}) -> {dst} ({size[0]}x{size[1]} @ {fps:.1f}fps, '
          f'{n_bytes / 1024 / 1024:.2f} MB)')
    if n_bytes > MAX_BYTES:
        print(f'WARNING: {dst} is over 5 MB')
