"""Step 5: relative camera pose (R, t) between scene frames from SIFT matches."""

import cv2
import numpy as np

RATIO = 0.75        # Lowe's ratio test
RANSAC_PX = 1.0     # max distance (px) from the epipolar line to count as inlier
MIN_MATCHES = 8     # 8-point algorithm needs at least 8 correspondences

_sift = cv2.SIFT_create()
_matcher = cv2.BFMatcher(cv2.NORM_L2)


def detect(frame: np.ndarray) -> tuple[list, np.ndarray | None]:
    """SIFT keypoints and their 128-d descriptors."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return _sift.detectAndCompute(gray, None)


def match_features(feat1: tuple, feat2: tuple, ratio: float = RATIO) -> tuple[np.ndarray, np.ndarray]:
    """Matched pixel coords (N, 2) in frame 1 and frame 2 (kNN + Lowe ratio test)."""
    (kp1, des1), (kp2, des2) = feat1, feat2
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
    good = [m for m, n in _matcher.knnMatch(des1, des2, k=2) if m.distance < ratio * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)
    return pts1, pts2


def estimate_pose(pts1: np.ndarray, pts2: np.ndarray, K: np.ndarray, D: np.ndarray | None = None) -> dict | None:
    """E (RANSAC, 5-point) -> recoverPose. None if too few matches or the pose is not trustworthy.

    Points are undistorted first (k2 = -0.31 is not negligible), keeping pixel
    units by re-projecting with K, so E describes the ideal pinhole camera.

    findEssentialMat is used instead of findFundamentalMat + E = K^T F K: knowing K
    leaves 5 unknowns instead of 7, which on 01_01 gave a more consistent t
    (0.90 vs 0.79 pair-to-pair agreement) and a more sideways t, as recorded.

    Rejected when under half the inliers triangulate in front of both cameras
    (cheirality): that is when recoverPose returns impossible ~180 deg rotations
    or noise (low texture, orbiting camera).
    Returns R (3x3), t (3x1 unit vector), E, F and match counts.
    """
    if len(pts1) < MIN_MATCHES:
        return None
    if D is not None:
        pts1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        pts2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)

    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, RANSAC_PX)
    if E is None or mask is None:
        return None
    E = E[:3]  # several solutions may come back stacked; the first is RANSAC's best
    # recoverPose picks the one of 4 (R, t) decompositions that puts points in front of both cameras
    n_front, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask.copy())
    inliers = int(mask.sum())
    if n_front < 0.5 * inliers:
        return None
    Kinv = np.linalg.inv(K)
    return {'R': R, 't': t, 'E': E, 'F': Kinv.T @ E @ Kinv,  # F for the report: E = K^T F K
            'inliers': inliers, 'in_front': int(n_front), 'matches': len(pts1),
            'pts1': pts1, 'pts2': pts2, 'mask': mask.ravel().astype(bool)}  # undistorted px


def rotation_deg(R: np.ndarray) -> float:
    """Rotation angle of R in degrees."""
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))


def scene_poses(frames: list[np.ndarray], K: np.ndarray, D: np.ndarray, step: int = 10) -> list[dict | None]:
    """Pose of frame t+step relative to frame t for every t (None where it failed).

    ponytail: step=10 tuned on 01_01 (consecutive frames move < 1 px median, too
    little for a stable E or for disparity); put it in the scene yaml if scenes differ.
    """
    feats = [detect(f) for f in frames]
    poses = []
    for t in range(len(frames) - step):
        pose = estimate_pose(*match_features(feats[t], feats[t + step]), K, D)
        if pose is not None:
            pose['frame'] = t
        poses.append(pose)
    return poses


if __name__ == '__main__':
    # Self-check on scene 01_01: R is a valid rotation, t is unit length, no 180 deg flips survive,
    # and the sideways first half of the video gives a mostly-x translation.
    from mypylibs.video import read_frames

    K = np.loadtxt('out/01_K.csv', delimiter=',')
    D = np.loadtxt('out/01_D.csv', delimiter=',')
    frames = read_frames('in/01_01.mp4')
    all_poses = scene_poses(frames, K, D)
    poses = [p for p in all_poses if p is not None]

    for p in poses:
        assert np.allclose(p['R'] @ p['R'].T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(p['R']), 1)
        assert np.isclose(np.linalg.norm(p['t']), 1)
        assert rotation_deg(p['R']) < 30, f'implausible rotation at frame {p["frame"]}'
    early = [p for p in poses if p['frame'] < 80]
    assert len(early) >= 70, 'sideways part of the video should almost always give a pose'
    assert all(abs(p['t'][0, 0]) > 0.9 for p in early), 'sideways slide should give t ~ (+-1, 0, 0)'

    inl = np.array([p['inliers'] / p['matches'] for p in poses])
    rot = np.array([rotation_deg(p['R']) for p in poses])
    print(f'{len(poses)}/{len(all_poses)} pairs kept: matches {np.median([p["matches"] for p in poses]):.0f} '
          f'(median), inlier ratio {inl.mean():.2f}, rotation per pair {np.median(rot):.2f} deg (median)')
    for p in poses[::20]:
        tx, ty, tz = p['t'].ravel()
        print(f'  frame {p["frame"]:3}: {p["inliers"]:4}/{p["matches"]:4} inliers, '
              f'rot {rotation_deg(p["R"]):.2f} deg, t=({tx:+.2f}, {ty:+.2f}, {tz:+.2f})')
