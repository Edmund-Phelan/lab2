"""Figures for report.tex, written to assets/latex/. Run after main.py (it reads out/).

    python -m mypylibs.report_figures

depth_examples.png: one depth pair per scene (rectified frame above, depth below).
The pairs are picked by hand from out/XX-YY_status.csv: high coverage and, for 02_01,
the mug part of the video rather than the window.
"""

import csv
import shutil
from os import path as opath

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mypylibs.depth import compute_depth
from mypylibs.pose import detect, estimate_pose, match_features
from mypylibs.runner import CM_PER_FRAME, ROOT_OUT
from mypylibs.video import read_frames

DST = opath.join('assets', 'latex')
EXAMPLES = [('01_01', 136), ('02_01', 67), ('03_01', 129)]  # (scene, first frame of the pair)


def partner(scene: str, first: int) -> int:
    """Second frame of the pair main.py used for `first` (from the status CSV)."""
    with open(opath.join(ROOT_OUT, scene.replace('_', '-') + '_status.csv')) as f:
        for row in csv.DictReader(f):
            if int(row['first_frame']) == first and row['status'] == 'OK':
                return int(row['second_frame'])
    raise ValueError(f'{scene}: frame {first} has no usable depth pair')


def depth_examples(dst: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 2.9), constrained_layout=True)
    cmap = plt.get_cmap('turbo_r').copy()
    cmap.set_bad('#303030')
    for col, (scene, i) in enumerate(EXAMPLES):
        cam = scene[:2]
        K = np.loadtxt(opath.join(ROOT_OUT, f'{cam}_K.csv'), delimiter=',')
        D = np.loadtxt(opath.join(ROOT_OUT, f'{cam}_D.csv'), delimiter=',')
        frames = read_frames(opath.join('in', f'{scene}.mp4'))
        j = partner(scene, i)
        pose = estimate_pose(*match_features(detect(frames[i]), detect(frames[j])), K, D)
        baseline = CM_PER_FRAME[scene] * (j - i)
        t_cm = pose['t'] * baseline / np.linalg.norm(pose['t'])
        image, depth = compute_depth(frames[i], frames[j], K, D, pose['R'], t_cm)

        low, high = np.nanpercentile(depth, [5, 95])
        axes[0, col].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title(f'{scene}: frames {i}$\\to${j}, $B$={baseline:.1f} cm', fontsize=7)
        im = axes[1, col].imshow(depth, cmap=cmap, vmin=low, vmax=high)
        cbar = fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label('depth (cm)', fontsize=6)
        for ax in axes[:, col]:
            ax.axis('off')
    fig.savefig(dst, dpi=200)
    plt.close(fig)
    print(f'wrote {dst}')


if __name__ == '__main__':
    depth_examples(opath.join(DST, 'depth_examples.png'))
    shutil.copy(opath.join(ROOT_OUT, '01_01_trajectory.png'), opath.join(DST, 'trajectory_01_01.png'))
    print(f'wrote {opath.join(DST, "trajectory_01_01.png")}')
