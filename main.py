import os
from os import path as opath

import numpy as np

from mypylibs.calibration import calibrate, find_all_corners, load_calibration_images

ROOT_IN = 'in'
ROOT_OUT = 'out'


def run_calibration(cam: int) -> None:
    """Calibrate camera `cam` from in/1.png ... in/N.png and save K, D to out/."""
    paths = load_calibration_images(ROOT_IN)
    if not paths:
        raise FileNotFoundError(f'No calibration frames in {ROOT_IN}/')
    image_points, size = find_all_corners(paths)

    res = calibrate(image_points, size)
    K, D = res['K'], res['D']

    print(f'Camera {cam:02}: {len(image_points)} images, {size[0]}x{size[1]}')
    print(f'  fx={K[0, 0]:.2f} fy={K[1, 1]:.2f} cx={K[0, 2]:.2f} cy={K[1, 2]:.2f} skew={K[0, 1]:.2f}')
    print(f'  k1={D[0]:.4f} k2={D[1]:.4f} p1={D[2]:.4f} p2={D[3]:.4f} k3={D[4]:.4f}')
    print(f'  reprojection error: mean={res["mean_error"]:.3f} px, rms={res["rms"]:.3f} px')
    for p, e in zip(paths, res['per_image_error']):
        print(f'    {p}: {e:.3f} px')

    os.makedirs(ROOT_OUT, exist_ok=True)
    np.savetxt(opath.join(ROOT_OUT, f'{cam:02}_K.csv'), K, delimiter=',', fmt='%.6f')
    np.savetxt(opath.join(ROOT_OUT, f'{cam:02}_D.csv'), D[None], delimiter=',', fmt='%.6f')


def main() -> None:
    run_calibration(1)


if __name__ == '__main__':
    main()
