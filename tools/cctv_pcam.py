"""Calibrate a session's player camera (pan-tilt-zoom: fixed centre, per-frame rotation + focal length).

Stage A (once per session and camera): the camera centre C and the frame shift to the TV clock. The players are the
calibration object: the TV view puts each player's feet on the court (rally_obs `foot`); the player camera's pose
detections give their ankles and shoulders. For a centre C, every sampled frame is solved for (R, f) with the best
assignment of the two most prominent people to TV players; C is the one whose frames all explain best (coarse grid,
then a joint refinement of C with every frame's R, f).
Stage B (every frame): R, f from the players, tracked frame to frame, then sharpened on the court lines (chamfer
fit of the projected court model to the white-line pixels).

Output: outputs/cctv/pcam_<view>_<session>.json {C, shift, stats}; per rally outputs/cctv_<view>/calib/<uid>.json
usage: python tools/cctv_pcam.py <session> <view p1|p2> [uid ...]   (dev uids default: outputs/cctv_pcam_dev_uids.txt)"""
import json, os, sys
from itertools import permutations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.players import load_pose  # noqa: E402

SIZE = (864, 486)
SHOULDER_Z = 1.40   # m, shoulder-midpoint height of a standing player (only a calibration anchor)
ANKLE_Z = 0.08
SIG = 6.0           # px


def project(C, rv, f, X):
    R = Rotation.from_rotvec(rv).as_matrix()
    Y = (np.asarray(X, float) - C) @ R.T
    z = np.maximum(Y[:, 2], 1e-3)
    return np.c_[f * Y[:, 0] / z + SIZE[0] / 2, f * Y[:, 1] / z + SIZE[1] / 2], Y[:, 2]


def look_at(C, target):
    """Rotation vector of a camera at C looking at target, image y down, world z up."""
    z = target - C
    z = z / np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x = x / max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.stack([x, y, z])).as_rotvec()


def people_2d(kps, min_conf=0.3):
    """Pose detections -> [(ankle_mid, shoulder_mid, height_px)] sorted by height (most prominent first)."""
    out = []
    for kp in kps:
        if min(kp[15, 2], kp[16, 2], kp[5, 2], kp[6, 2]) < min_conf:
            continue
        a, s = kp[[15, 16], :2].mean(0), kp[[5, 6], :2].mean(0)
        out.append((a, s, float(a[1] - kp[:, 1][kp[:, 2] > min_conf].min())))
    return sorted(out, key=lambda p: -p[2])


def solve_frame(C, img, world, x0=None):
    """R (rotvec), f for one frame from 2D-3D points; -> (rv, f, rms px)."""
    if x0 is None:
        x0 = np.r_[look_at(C, world.mean(0)), np.log(900.0)]

    def res(x):
        uv, z = project(C, x[:3], np.exp(x[3]), world)
        return np.r_[((uv - img) / SIG).ravel(), np.minimum(z - 0.5, 0) * 100]
    r = least_squares(res, x0, loss="soft_l1", max_nfev=60)
    uv, _ = project(C, r.x[:3], np.exp(r.x[3]), world)
    return r.x, float(np.sqrt(np.mean(np.sum((uv - img) ** 2, 1))))


def frame_options(feet, ppl):
    """Candidate (img (4,2), world (4,3)) assignments of the two most prominent people to two TV players."""
    ok = [j for j in range(len(feet)) if np.isfinite(feet[j]).all()]
    if len(ppl) < 2 or len(ok) < 2:
        return []
    img = np.array([ppl[0][0], ppl[0][1], ppl[1][0], ppl[1][1]])
    out = []
    for a, b in permutations(ok, 2):
        g = [np.r_[feet[a], ANKLE_Z], np.r_[feet[a], SHOULDER_Z], np.r_[feet[b], ANKLE_Z], np.r_[feet[b], SHOULDER_Z]]
        out.append((img, np.array(g)))
    return out


def samples(uids, view, shift, every=6):
    """[(options per frame)] over the dev rallies, player-camera frame = TV frame + shift."""
    S = []
    for u in uids:
        ob = os.path.join(ROOT, "outputs", "rally_obs", "cctv", u + ".npz")
        pp = os.path.join(ROOT, "outputs", "cctv_" + view, "pose1280", u + ".npz")
        if not (os.path.exists(ob) and os.path.exists(pp)):
            continue
        foot = np.load(ob)["foot"]
        kps = load_pose(pp)
        for t in range(0, len(foot), every):
            k = t + shift
            if 0 <= k < len(kps):
                o = frame_options(foot[t], people_2d(kps[k]))
                if o:
                    S.append(o)
    return S


def score_centre(C, S):
    """Median over frames of the best assignment's rms (px) for camera centre C."""
    best = []
    for opts in S:
        best.append(min(solve_frame(C, img, w)[1] for img, w in opts))
    return float(np.median(best)), best


def main():
    sess, view = sys.argv[1], sys.argv[2]
    uids = sys.argv[3:] or open(os.path.join(ROOT, "outputs", "cctv_pcam_dev_uids.txt")).read().split()
    rng = np.random.default_rng(0)
    S0 = samples(uids, view, 0, every=10)
    sub = [S0[i] for i in rng.choice(len(S0), min(40, len(S0)), replace=False)]
    print(view, len(S0), "usable frames; grid search on", len(sub), flush=True)
    grid = []
    for x in np.arange(-14, 21, 3.0):
        for y in np.arange(-10, 24, 3.0):
            if -1 < x < 7 and -1 < y < 14.4:
                continue  # not on the court
            for z in (1.0, 3.0, 6.0):
                C = np.array([x, y, z])
                grid.append((score_centre(C, sub[:15])[0], C))
    grid.sort(key=lambda g: g[0])
    print("best grid centres:", [(round(s, 1), c.tolist()) for s, c in grid[:5]], flush=True)
    # refine the best few on more frames
    best = None
    for s, C in grid[:5]:
        r = least_squares(lambda c: np.array(score_centre(c, sub)[1]) / SIG, C, diff_step=0.05, max_nfev=30)
        sc = score_centre(r.x, sub)[0]
        print(" refined", C.tolist(), "->", r.x.round(2).tolist(), "median rms %.1f px" % sc, flush=True)
        if best is None or sc < best[0]:
            best = (sc, r.x)
    C = best[1]
    # frame shift to the TV clock
    shifts = []
    for sh in range(-12, 13, 2):
        Ssh = samples(uids, view, sh, every=10)
        Ssh = [Ssh[i] for i in rng.choice(len(Ssh), min(60, len(Ssh)), replace=False)]
        shifts.append((score_centre(C, Ssh)[0], sh))
    shifts.sort()
    print("shift scores:", shifts[:5], flush=True)
    out = {"C": C.tolist(), "shift": int(shifts[0][1]), "median_rms_px": round(best[0], 2), "frames": len(S0)}
    json.dump(out, open(os.path.join(ROOT, "outputs", "cctv", f"pcam_{view}_{sess}.json"), "w"), indent=1)
    print("RESULT", view, out, flush=True)


if __name__ == "__main__":
    main()
