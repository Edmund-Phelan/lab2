"""Run the whole pipeline for every camera found in in/ (see mypylibs/runner.py)."""

import os

from mypylibs.runner import ROOT_IN, ROOT_OUT, cameras, run_calibration, run_scene, scenes


def main() -> None:
    os.makedirs(ROOT_OUT, exist_ok=True)
    cams = cameras()
    if not cams:
        raise FileNotFoundError(f'No XX_calibration.mp4 in {ROOT_IN}/')
    for cam in cams:
        print(f'\n=== Camera {cam} ===')
        K, D = run_calibration(cam)
        for video in scenes(cam):
            run_scene(video, K, D)


if __name__ == '__main__':
    main()
