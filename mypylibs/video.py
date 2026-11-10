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


def read_frames(path: str) -> list:
    """All frames of an (already pre-processed) video as BGR arrays."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {path}')
    frames = []
    ok, frame = cap.read()
    while ok:
        frames.append(frame)
        ok, frame = cap.read()
    cap.release()
    return frames


def preprocess_video(src: str, dst: str, every: int = 1) -> None:
    """Downscale src to fit 640x360 and write it to dst as mp4, keeping every `every`-th frame.

    Dropping frames (to get under 5 MB) doesn't change the pixels of the kept ones,
    so the intrinsics stay valid. Only do it for calibration videos: scenes need
    the small inter-frame motion.
    """
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

    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps / every, size)
    i = 0
    while ok:
        if i % every == 0:
            writer.write(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        ok, frame = cap.read()
        i += 1

    cap.release()
    writer.release()

    n_bytes = opath.getsize(dst)
    print(f'{src} ({w}x{h}) -> {dst} ({size[0]}x{size[1]} @ {fps / every:.1f}fps, '
          f'{n_bytes / 1024 / 1024:.2f} MB)')
    if n_bytes > MAX_BYTES:
        print(f'WARNING: {dst} is over 5 MB')


if __name__ == '__main__':
    # Usage: python -m mypylibs.video <raw_video> <in/XX_calibration.mp4 | in/XX_YY.mp4> [every]
    import sys
    if len(sys.argv) < 3:
        sys.exit('usage: python -m mypylibs.video <src> <dst> [every]')
    preprocess_video(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
