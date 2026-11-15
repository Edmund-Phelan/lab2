"""Step 5: relative camera pose (R, t) between scene frames from SIFT matches."""

import cv2
import numpy as np

RATIO = 0.75        # Lowe's ratio test
RANSAC_PX = 1.0     # max distance (px) from the epipolar line to count as inlier
MIN_MATCHES = 8     # 8-point algorithm needs at least 8 correspondences
MAX_DEPTH = 1e4     # in baselines; recoverPose's default (50) drops most points of a small-baseline pair
MIN_PARALLAX_PX = 3 # below this median motion t is noise (01_01 frames 0-2 gave t ~ forward)
MIN_AGREEMENT = 0.9 # share of inliers whose rotation-free flow must point the same way (see estimate_pose)
MIN_FLOW_PX = 10   # median feature motion wanted between the two frames of a depth pair
MAX_GAP = 90        # frames (3 s); give up looking for a partner frame after this

# Default contrastThreshold (0.04) finds only ~250 keypoints on 640x360 frames; 0.02 roughly doubles it
_sift = cv2.SIFT_create(contrastThreshold=0.02)
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

    Rejected when the features barely moved (no parallax to recover t from), or under half the inliers triangulate in front of both cameras
    (cheirality): that is when recoverPose returns impossible ~180 deg rotations
    or noise (low texture, orbiting camera). distanceThresh is raised to MAX_DEPTH:
    at the default of 50 baselines, points further away (most of a 1 cm-baseline
    scene) don't count as in front, and good poses were rejected. Also rejected when
    the rotation-free flow doesn't point one way (an ambiguous pan vs. slide).
    Returns R (3x3), t (3x1 unit vector), E, F and match counts.
    """
    if len(pts1) < MIN_MATCHES:
        return None
    if D is not None:
        pts1 = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
        pts2 = cv2.undistortPoints(pts2.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)

    if median_flow(pts1, pts2) < MIN_PARALLAX_PX:
        return None

    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, RANSAC_PX)
    if E is None or mask is None:
        return None
    E = E[:3]  # several solutions may come back stacked; the first is RANSAC's best
    # recoverPose picks the one of 4 (R, t) decompositions that puts points in front of both cameras
    n_front, R, t, _, _ = cv2.recoverPose(E, pts1, pts2, K, distanceThresh=MAX_DEPTH, mask=mask.copy())
    inliers = int(mask.sum())
    if n_front < 0.5 * inliers:
        return None
    Kinv = np.linalg.inv(K)

    # With little parallax a small pan explains the flow about as well as a slide, and
    # recoverPose returns (R, t) pairs with the wrong pan and t flipped (on 01_01, t jumped
    # between +x and -x while every feature moved left). For a sideways slide, undoing R
    # should leave flow that all points one way; a wrong R leaves it scattered.
    inl = mask.ravel().astype(bool)
    derotated = cv2.perspectiveTransform(pts2[inl].reshape(-1, 1, 2), K @ R.T @ Kinv).reshape(-1, 2)
    flow = derotated - pts1[inl]
    if np.mean(flow @ np.median(flow, axis=0) > 0) < MIN_AGREEMENT:
        return None

    return {'R': R, 't': t, 'E': E, 'F': Kinv.T @ E @ Kinv,  # F for the report: E = K^T F K
            'inliers': inliers, 'in_front': int(n_front), 'matches': len(pts1),
            'pts1': pts1, 'pts2': pts2, 'mask': mask.ravel().astype(bool)}  # undistorted px


def rotation_deg(R: np.ndarray) -> float:
    """Rotation angle of R in degrees."""
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))


def median_flow(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """Median distance (px) the matched features moved."""
    return float(np.median(np.linalg.norm(pts2 - pts1, axis=1))) if len(pts1) else 0.0


def depth_pairs(frames: list[np.ndarray], K: np.ndarray, D: np.ndarray,
                min_flow: float = MIN_FLOW_PX, max_gap: int = MAX_GAP) -> list[dict | None]:
    """For every frame t, the pose of the first later frame whose features moved >= min_flow px.

    A fixed gap doesn't suit every video: 01_01 moves ~20 px in 10 frames, 03_01
    (far building) only 1-3 px, too little parallax for a stable E or disparity.
    Each pose gets 'frame' (t) and 'partner' (t + gap); None where no partner was
    found within max_gap frames or the pose was rejected.
    """
    feats = [detect(f) for f in frames]
    poses, j = [], 1
    for t in range(len(frames) - 1):
        # The camera keeps moving the same way, so frame t+1's partner is rarely before frame t's
        j = max(j - 2, t + 1)
        pose = None
        while j < min(len(frames), t + max_gap + 1):
            pts1, pts2 = match_features(feats[t], feats[j])
            if median_flow(pts1, pts2) >= min_flow:
                pose = estimate_pose(pts1, pts2, K, D)
                break
            j += 1
        if pose is not None:
            pose['frame'], pose['partner'] = t, j
        poses.append(pose)
    return poses


if __name__ == '__main__':
    # Self-check on scene 01_01: R is a valid rotation, t is unit length, no 180 deg flips survive,
    # and the sideways slide gives a mostly-x translation, the same way every time.
    from mypylibs.video import read_frames

    K = np.loadtxt('out/01_K.csv', delimiter=',')
    D = np.loadtxt('out/01_D.csv', delimiter=',')
    frames = read_frames('in/01_01.mp4')
    all_poses = depth_pairs(frames, K, D)
    poses = [p for p in all_poses if p is not None]

    for p in poses:
        assert np.allclose(p['R'] @ p['R'].T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(p['R']), 1)
        assert np.isclose(np.linalg.norm(p['t']), 1)
        assert rotation_deg(p['R']) < 30, f'implausible rotation at frame {p["frame"]}'
    assert len(poses) >= 60, 'too few depth pairs kept on a sideways slide'
    sideways = np.mean([p['t'][0, 0] < -0.85 for p in poses])
    assert sideways >= 0.9, f'only {sideways:.0%} of pairs give t ~ (-1, 0, 0) (camera slides right)'

    inl = np.array([p['inliers'] / p['matches'] for p in poses])
    rot = np.array([rotation_deg(p['R']) for p in poses])
    print(f'{len(poses)}/{len(all_poses)} pairs kept: matches {np.median([p["matches"] for p in poses]):.0f} '
          f'(median), inlier ratio {inl.mean():.2f}, rotation per pair {np.median(rot):.2f} deg (median)')
    for p in poses[::20]:
        tx, ty, tz = p['t'].ravel()
        print(f'  frame {p["frame"]:3}: {p["inliers"]:4}/{p["matches"]:4} inliers, '
              f'rot {rotation_deg(p["R"]):.2f} deg, t=({tx:+.2f}, {ty:+.2f}, {tz:+.2f})')
