"""Scene video overlay: SIFT features and how far each moved since the previous frame.

Usage (from the repo root):
    python -m mypylibs.visualize_scene <scene_video> [out_video]

Each matched feature is a dot with a line back to where it was GAP frames earlier,
coloured by how far it moved. With a sideways camera, near objects move more
than far ones, so the colours preview the depth map (warm = near, cool = far).
"""

import sys

import cv2
import numpy as np

GAP = 3          # frames between the matched pair (1 = consecutive; 3 makes lines visible)
RATIO = 0.75     # Lowe's ratio test for SIFT matches


def label(img: np.ndarray, text: str, org: tuple[int, int], color=(255, 255, 255), scale: float = 0.5) -> None:
    """Text on a dark box so it reads on any background."""
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def match(sift, matcher, kd_a, kd_b) -> tuple[np.ndarray, np.ndarray]:
    """Matched point coords (N, 2) in frame a and frame b."""
    (kp_a, des_a), (kp_b, des_b) = kd_a, kd_b
    if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    good = [m for m, n in matcher.knnMatch(des_a, des_b, k=2) if m.distance < RATIO * n.distance]
    pa = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pb = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pa, pb


def make_video(video: str, dst: str) -> None:
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()

    sift = cv2.SIFT_create()
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    feats = [sift.detectAndCompute(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), None) for f in frames]

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    stats = []
    for t in range(GAP, len(frames)):
        prev_pts, pts = match(sift, matcher, feats[t - GAP], feats[t])
        out = frames[t].copy()
        if len(pts):
            # Drop wrong matches (repeated text / checker squares) that don't fit
            # one camera motion: RANSAC fundamental matrix inliers, as in step 5
            if len(pts) >= 8:
                _, mask = cv2.findFundamentalMat(prev_pts, pts, cv2.FM_RANSAC, 1.0, 0.99)
                if mask is not None:
                    keep = mask.ravel() == 1
                    prev_pts, pts = prev_pts[keep], pts[keep]
            d = pts - prev_pts
            mag = np.linalg.norm(d, axis=1)

            top = max(np.percentile(mag, 95), 1e-6) if len(mag) else 1
            colors = cv2.applyColorMap(np.uint8(np.clip(mag / top, 0, 1) * 255).reshape(-1, 1),
                                       cv2.COLORMAP_TURBO).reshape(-1, 3)
            for (x0, y0), (x1, y1), c in zip(prev_pts, pts, colors):
                c = tuple(int(v) for v in c)
                cv2.line(out, (round(x0), round(y0)), (round(x1), round(y1)), c, 1, cv2.LINE_AA)
                cv2.circle(out, (round(x1), round(y1)), 2, c, -1, cv2.LINE_AA)
            dx, dy = (np.median(d[:, 0]), np.median(d[:, 1])) if len(d) else (0, 0)
            stats.append((len(pts), dx, dy))
            label(out, f'{len(pts)} SIFT matches   median motion dx={dx:+.1f}px dy={dy:+.1f}px', (10, 22))
        label(out, f'frame {t - GAP} -> {t}', (10, 44))
        label(out, 'colour = distance moved: warm = near, cool = far', (10, h - 12), scale=0.45)
        writer.write(out)
    writer.release()

    n, dx, dy = np.array(stats).T
    print(f'{len(frames)} frames; per pair (gap {GAP}): {n.mean():.0f} matches on average, '
          f'median dx={np.median(dx):+.1f}px dy={np.median(dy):+.1f}px; wrote {dst}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    make_video(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'out/01_01_motion.mp4')
