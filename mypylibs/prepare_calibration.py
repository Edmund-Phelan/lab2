"""Shrink a raw calibration video and extract 20 calibration frames from it.

Usage (from the repo root):
    python -m mypylibs.prepare_calibration <raw_video> [camera_id]

Writes in/<XX>_calibration.mp4 and in/1.png ... in/20.png.
"""

import sys
from os import path as opath

from mypylibs.frames import extract_calibration_frames
from mypylibs.video import preprocess_video

ROOT_IN = 'in'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cam = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    small = opath.join(ROOT_IN, f'{cam:02}_calibration.mp4')
    preprocess_video(sys.argv[1], small)
    extract_calibration_frames(small, ROOT_IN, 20)
