"""Player-camera (pan-tilt-zoom) calibration, chained.

A player camera stays at one centre C for the whole session and only turns and zooms. Within a rally, consecutive
frames are related by a rotation + zoom homography (K_b dR K_a^-1), measured from SIFT matches on the static hall
(hundreds of inliers per pair). Given the first frame's focal length f0 this chain fixes every frame's rotation
relative to the first and its focal length, so a rally has only 4 unknowns left (first-frame rotation R0, f0) plus
the session's C and the frame shift to the TV clock. Those are pinned by:
  - the players: the TV view puts their feet on the court (rally_obs foot); the player camera sees their ankles;
  - the court lines: two-way chamfer (precision + recall) on frames that show the floor.

Stages (python tools/pcam_calib.py <stage> <view p1|p2> [uid ...]):
  match   SIFT chain matches per rally -> outputs/cctv_<view>/pcam/match/<uid>.npz
  centre  grid search of C on dev rallies (players + lines) -> outputs/cctv/pcam_<view>_<session>.json
  rally   per rally R0, f0, shift with C fixed, then per-frame line refinement -> outputs/cctv_<view>/pcam/calib/<uid>.npz"""
import json, os, sys
from multiprocessing import Pool

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import Camera  # noqa: E402
from birdseye_v2.courtfit import line_mask, model_points  # noqa: E402
from birdseye_v2.players import load_pose  # noqa: E402

SIZE = (864, 486)
CX, CY = SIZE[0] / 2, SIZE[1] / 2
ANKLE_Z = 0.08
MAXM = 80  # matches kept per frame pair


def clip(view, uid):
    return os.path.join(ROOT, "data", "cctv", "clips_" + view, uid + ".mp4")


def pdir(view, *a):
    d = os.path.join(ROOT, "outputs", "cctv_" + view, "pcam", *a)
    os.makedirs(d, exist_ok=True)
    return d


# ------------------------------------------------------------------ chain -----------------------------------------
def pose_path(view, uid):
    for d in ("pose1280", "pose"):
        p = os.path.join(ROOT, "outputs", "cctv_" + view, d, uid + ".npz")
        if os.path.exists(p):
            return p
    return None


def static_mask(people):
    """Where the hall can be matched: not the broadcast watermark / score overlays (they stay put on screen while
    the camera turns) and not the people (they move on their own)."""
    m = np.full((SIZE[1], SIZE[0]), 255, np.uint8)
    m[:100, 660:] = 0  # watermark
    for kp in people:
        v = kp[:, 2] > 0.2
        if v.sum() < 3:
            continue
        x0, y0 = kp[v, :2].min(0); x1, y1 = kp[v, :2].max(0)
        pw, ph = x1 - x0, y1 - y0
        m[max(0, int(y0 - 0.35 * ph)):int(y1 + 0.15 * ph), max(0, int(x0 - 0.6 * pw)):int(x1 + 0.6 * pw)] = 0  # + racket reach
    return m


def match(args):
    view, uid = args
    dst = os.path.join(pdir(view, "match"), uid + ".npz")
    if os.path.exists(dst):
        return uid, "done"
    cv2.setNumThreads(1)
    sift = cv2.SIFT_create(1000)
    bf = cv2.BFMatcher()
    cap = cv2.VideoCapture(clip(view, uid))
    pp = pose_path(view, uid)
    kps = load_pose(pp) if pp else []
    prev, A, B, n = None, [], [], 0
    rng = np.random.default_rng(0)
    while True:
        ok, im = cap.read()
        if not ok:
            break
        kp, de = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), static_mask(kps[n] if n < len(kps) else []))
        pa = pb = np.zeros((0, 2), np.float32)
        if prev is not None and de is not None and prev[1] is not None and len(kp) >= 8 and len(prev[0]) >= 8:
            m = bf.knnMatch(prev[1], de, k=2)
            good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
            if len(good) >= 8:
                a = np.float32([prev[0][g.queryIdx].pt for g in good])
                b = np.float32([kp[g.trainIdx].pt for g in good])
                H, mk = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
                if mk is not None:
                    k = np.nonzero(mk.ravel())[0]
                    k = k if len(k) <= MAXM else rng.choice(k, MAXM, replace=False)
                    pa, pb = a[k], b[k]
        if prev is not None:
            A.append(pa); B.append(pb)
        prev = (kp, de)
        n += 1
    cnt = np.array([len(a) for a in A], int)
    np.savez_compressed(dst, n=n, cnt=cnt, A=np.concatenate(A) if A else np.zeros((0, 2)), B=np.concatenate(B) if B else np.zeros((0, 2)))
    return uid, f"{n} frames, inliers median {np.median(cnt) if len(cnt) else 0:.0f}, min {cnt.min() if len(cnt) else 0}"


def load_match(view, uid):
    z = np.load(os.path.join(pdir(view, "match"), uid + ".npz"))
    off = np.r_[0, np.cumsum(z["cnt"])]
    return int(z["n"]), [(z["A"][off[i]:off[i + 1]], z["B"][off[i]:off[i + 1]]) for i in range(len(z["cnt"]))]


def _rot_res(x, fa, a, b):
    dR = Rotation.from_rotvec(x[:3]).as_matrix()
    fb = fa * np.exp(x[3])
    d = np.c_[(a[:, 0] - CX) / fa, (a[:, 1] - CY) / fa, np.ones(len(a))] @ dR.T
    return np.r_[(fb * d[:, 0] / d[:, 2] + CX - b[:, 0]), (fb * d[:, 1] / d[:, 2] + CY - b[:, 1])]


def chain(pairs, f0):
    """-> Rrel (n,3,3) with R_t = Rrel[t] @ R0 (world->camera), f (n,), per-pair rms px."""
    n = len(pairs) + 1
    Rrel = np.zeros((n, 3, 3)); Rrel[0] = np.eye(3)
    f = np.zeros(n); f[0] = f0
    rms = np.zeros(n - 1)
    x = np.zeros(4)
    for i, (a, b) in enumerate(pairs):
        if len(a) < 6:
            Rrel[i + 1], f[i + 1], rms[i] = Rrel[i], f[i], np.nan
            continue
        r = least_squares(_rot_res, x, args=(f[i], a, b), loss="huber", f_scale=1.5, max_nfev=30)
        Rrel[i + 1] = Rotation.from_rotvec(r.x[:3]).as_matrix() @ Rrel[i]
        f[i + 1] = f[i] * np.exp(r.x[3])
        rms[i] = np.sqrt(np.median(r.fun ** 2))
        x = r.x  # smooth motion: start the next pair from this one
    return Rrel, f, rms


# ------------------------------------------------------------------ observations ----------------------------------
def people(kps, min_conf=0.3):
    """Pose detections -> [(ankle_mid px, height px)], tallest first."""
    out = []
    for kp in kps:
        if min(kp[15, 2], kp[16, 2]) < min_conf:
            continue
        vis = kp[:, 2] > min_conf
        out.append((kp[[15, 16], :2].mean(0), float(kp[vis, 1].max() - kp[vis, 1].min())))
    return sorted(out, key=lambda p: -p[1])


def player_obs(view, uid, shift, every=2):
    """[(pcam frame k, ankle px (m,2), TV feet (j,3))] for pcam frames k = TV frame + shift."""
    ob = os.path.join(ROOT, "outputs", "rally_obs", "cctv", uid + ".npz")
    pp = pose_path(view, uid)
    if not (os.path.exists(ob) and pp):
        return []
    foot = np.load(ob)["foot"]
    kps = load_pose(pp)
    out = []
    for t in range(0, len(foot), every):
        k = t + shift
        if not 0 <= k < len(kps):
            continue
        ok = [j for j in range(len(foot[t])) if np.isfinite(foot[t][j]).all()]
        pp_ = [p for p in people(kps[k]) if p[1] > 60][:3]
        if ok and pp_:
            out.append((k, np.array([p[0] for p in pp_]), np.array([np.r_[foot[t][j], ANKLE_Z] for j in ok])))
    return out


def line_frames(view, uid, n_max=14):
    """Frames that show floor lines: [(k, dt, mask pts)]."""
    cap = cv2.VideoCapture(clip(view, uid))
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for k in np.linspace(0, N - 1, 3 * n_max).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, im = cap.read()
        if not ok:
            continue
        m = line_mask(im, wide=True)
        if (m > 0).sum() < 1500:
            continue
        dt = cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3)
        pts = np.stack(np.nonzero(m)[::-1], 1)
        pts = pts[np.random.default_rng(int(k)).choice(len(pts), min(1500, len(pts)), replace=False)]
        out.append((int(k), dt, pts, int((m > 0).sum())))
    out.sort(key=lambda o: -o[3])
    return out[:n_max]


MP = model_points(0.1)


K1 = [0.0]  # radial term of the current camera (set from the clicked calibration)


def _line_canvas(R, f, C):
    cam = Camera(f, cv2.Rodrigues(R)[0].ravel(), -R @ C, SIZE, (K1[0], 0.0))
    uv = cam.project(MP)
    Y = (MP - C) @ R.T
    z = Y[:, 2].copy()
    r2 = (Y[:, 0] ** 2 + Y[:, 1] ** 2) / np.maximum(z, 1e-6) ** 2
    z[r2 > 1.6 * (np.hypot(CX, CY) / f) ** 2] = -1  # far outside the picture: the distortion polynomial folds it back in
    return uv, z


def line_score(R, f, C, dt, pts, tol=4.0):
    """(precision, recall): share of visible model samples on paint; share of paint pixels near a model sample."""
    uv, z = _line_canvas(R, f, C)
    vis = (z > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < SIZE[0] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE[1] - 1)
    if vis.sum() < 40:
        return 0.0, 0.0
    prec = float((dt[uv[vis, 1].astype(int), uv[vis, 0].astype(int)] < tol).mean())
    canvas = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    q = np.round(uv[vis]).astype(int)
    canvas[q[:, 1], q[:, 0]] = 255
    canvas = cv2.dilate(canvas, np.ones((2 * int(tol) + 5, 2 * int(tol) + 5), np.uint8))  # model samples 10 cm apart
    rec = float((canvas[pts[:, 1], pts[:, 0]] > 0).mean())
    return prec, rec


# ------------------------------------------------------------------ gauge fit -------------------------------------
def project(R, f, C, X):
    Y = (X - C) @ R.T
    z = Y[:, 2]
    return np.c_[f * Y[:, 0] / np.maximum(z, 1e-3) + CX, f * Y[:, 1] / np.maximum(z, 1e-3) + CY], z


def look_at(C, target):
    z = target - C; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= max(np.linalg.norm(x), 1e-9)
    return np.stack([x, np.cross(z, x), z])


def pack_obs(obs):
    """Observations as padded arrays: frame k (m,), ankles (m,3,2) nan-padded, TV feet (m,4,3) nan-padded."""
    m = len(obs)
    K = np.array([o[0] for o in obs], int)
    A = np.full((m, 3, 2), np.nan); F = np.full((m, 4, 3), np.nan)
    for i, (k, ank, feet) in enumerate(obs):
        A[i, :len(ank)] = ank[:3]; F[i, :len(feet)] = feet[:4]
    return K, A, F


def gauge_res(x, C, Rrel, f, ob, sig=8.0):
    """x = (rotvec R0, log f0 scale). Per observed ankle: px distance to the nearest projected TV foot of that frame."""
    K, A, F = ob
    R0 = Rotation.from_rotvec(x[:3]).as_matrix()
    Rk = Rrel[K] @ R0                                   # (m,3,3)
    Y = np.einsum("mij,mpj->mpi", Rk, F - C)            # (m,4,3) camera coords
    fk = f[K] * np.exp(x[3])
    z = Y[..., 2]
    uv = np.stack([fk[:, None] * Y[..., 0] / np.maximum(z, 1e-3) + CX, fk[:, None] * Y[..., 1] / np.maximum(z, 1e-3) + CY], -1)
    uv[~(z > 0.3)] = 1e4
    d = np.linalg.norm(A[:, :, None] - uv[:, None], axis=-1)  # (m,3,4)
    d = np.where(np.isnan(d), np.inf, d).min(2)                # (m,3)
    d = d[np.isfinite(d) & ~np.isnan(A[..., 0])]
    return np.minimum(d, 400) / sig


def fit_gauge(C, Rrel, f, ob):
    """Best first-frame rotation (and a small f0 correction) for centre C: multi-start from look-ats that put a TV
    player under an observed ankle, robust refine."""
    K, A, F = ob
    starts = []
    for i in np.linspace(0, len(K) - 1, 8).astype(int):
        k = K[i]
        for j in range(4):
            if np.isnan(F[i, j, 0]):
                continue
            L = look_at(C, F[i, j])
            ray = np.array([(A[i, 0, 0] - CX) / f[k], (A[i, 0, 1] - CY) / f[k], 1.0]); ray /= np.linalg.norm(ray)
            fix = Rotation.align_vectors([ray], [[0, 0, 1.0]])[0].as_matrix()
            starts.append(np.r_[Rotation.from_matrix(Rrel[k].T @ fix @ L).as_rotvec(), 0.0])
    best = min(starts, key=lambda s: np.median(gauge_res(s, C, Rrel, f, ob)))
    r = least_squares(gauge_res, best, args=(C, Rrel, f, ob), loss="cauchy", f_scale=1.0, max_nfev=80,
                      bounds=([-np.inf] * 3 + [-0.3], [np.inf] * 3 + [0.3]))
    return r.x, float(np.median(r.fun) * 8.0)


def cam_at(x, C, Rrel, f, k):
    return Rrel[k] @ Rotation.from_rotvec(x[:3]).as_matrix(), f[k] * np.exp(x[3])


def line_res(R, f, C, dt, tr=12.0):
    """Truncated distance (px) from each visible projected court-line sample to the nearest paint pixel."""
    uv, z = _line_canvas(R, f, C)
    vis = (z > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < SIZE[0] - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < SIZE[1] - 1)
    r = np.full(len(MP), tr, np.float32)
    if vis.any():
        m = cv2.remap(dt, uv[vis, 0].astype(np.float32).reshape(-1, 1), uv[vis, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR)
        r[vis] = np.minimum(m.ravel(), tr)
    return r


def refine_joint(C, x, Rrel, f, ob, LF, fix_c=False, w_line=0.5, truncs=(40.0, 20.0, 8.0), verbose=False):
    """C (unless fixed), first-frame rotation and f scale on players + court lines together, coarse to fine."""
    def res(p, tr):
        Cc = C if fix_c else p[4:7]
        out = [gauge_res(p[:4], Cc, Rrel, f, ob)]
        for k, dt, pts, _ in LF:
            R, fk = cam_at(p[:4], Cc, Rrel, f, k)
            out.append(w_line * line_res(R, fk, Cc, dt, tr) / 4.0)
        return np.concatenate(out)
    p = np.r_[x, C] if not fix_c else np.asarray(x, float)
    for tr in truncs:
        r = least_squares(res, p, args=(tr,), loss="cauchy", f_scale=1.0, max_nfev=200, diff_step=1e-3)
        if verbose:
            print("  trunc", tr, "cost %.1f -> %.1f" % (0.5 * np.sum(np.log1p(res(p, tr) ** 2)), r.cost), "nfev", r.nfev, "C", np.round(r.x[4:7], 2), flush=True)
        p = r.x
    return (p[4:7] if not fix_c else C), p[:4]


def fit_frame_lines(R, f, C, dt, truncs=(30.0, 15.0, 6.0)):
    """One frame's rotation + focal length on the court lines (C fixed), coarse to fine -> (R, f, rms px of
    visible samples within 6 px, precision)."""
    def res(p, tr):
        return line_res(Rotation.from_rotvec(p[:3]).as_matrix() @ R, f * np.exp(p[3]), C, dt, tr)
    p = np.zeros(4)
    for tr in truncs:
        p = least_squares(res, p, args=(tr,), loss="soft_l1", f_scale=3.0, max_nfev=100, diff_step=1e-4).x
    R2, f2 = Rotation.from_rotvec(p[:3]).as_matrix() @ R, f * np.exp(p[3])
    r = res(p, 50.0)
    return R2, f2, float(np.mean(r < 3.0)), float(np.median(r))


def gauge_line_search(C, x, Rrel, f, LF, span=(6.0, 6.0, 3.0, 0.35), steps=(9, 9, 5, 7)):
    """Grid over small turns (pan, tilt, roll in degrees, in the first frame's camera axes) and f scale around a
    gauge x; keep the one whose court lines land on paint best over the rally's line frames (mean precision x
    recall) -> (x, score)."""
    R0 = Rotation.from_rotvec(x[:3]).as_matrix()
    best = (-1, x)
    for a in np.linspace(-span[0], span[0], steps[0]):
        for b in np.linspace(-span[1], span[1], steps[1]):
            for c in np.linspace(-span[2], span[2], steps[2]):
                T = Rotation.from_euler("yxz", [a, b, c], degrees=True).as_matrix()
                rv = Rotation.from_matrix(T @ R0).as_rotvec()
                for s in np.linspace(-span[3], span[3], steps[3]):
                    xx = np.r_[rv, x[3] + s]
                    sc = np.mean([np.prod(line_score(*cam_at(xx, C, Rrel, f, k), C, dt, pts)) for k, dt, pts, _ in LF])
                    if sc > best[0]:
                        best = (sc, xx)
    return best[1], best[0]


def chain_from(pairs, a, fa):
    """Chain anchored at frame a: Rrel[a] = I, f[a] = fa; forwards with the pairs, backwards with them inverted."""
    n = len(pairs) + 1
    fw, ffw, _ = chain(pairs[a:], fa)
    bw, fbw, _ = chain([(q, p) for p, q in pairs[:a][::-1]], fa)
    Rrel = np.concatenate([bw[1:][::-1], fw]); f = np.concatenate([fbw[1:][::-1], ffw])
    assert len(Rrel) == n
    return Rrel, f


def line_evidence(view, uid, every=3):
    """Frame -> (dt, mask pts, paint px) for every `every`-th frame that shows floor lines."""
    cap = cv2.VideoCapture(clip(view, uid))
    out, k = {}, 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k % every == 0:
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            if (cv2.inRange(hsv, (35, 60, 50), (95, 255, 255)) > 0).mean() > 0.15:
                m = line_mask(im, wide=True)
                npx = int((m > 0).sum())
                if npx > 1500:
                    dt = cv2.distanceTransform((m == 0).astype(np.uint8), cv2.DIST_L2, 3)
                    pts = np.stack(np.nonzero(m)[::-1], 1)
                    pts = pts[np.random.default_rng(k).choice(len(pts), min(1500, len(pts)), replace=False)]
                    out[k] = (dt, pts, npx)
        k += 1
    return out


_REFS = {}


def refs(view, sess):
    """The hand-clicked frames of this camera and session: (R, f, SIFT keypoints, descriptors) each."""
    key = (view, sess)
    if key not in _REFS:
        S = json.load(open(os.path.join(ROOT, "outputs", "cctv", f"pcam_{view}_{sess}.json")))
        sift = cv2.SIFT_create(3000)
        out = []
        for fid, v in S["frames"].items():
            im = cv2.imread(os.path.join(ROOT, "outputs", "cctv", "pcam_clicks", "img", fid + ".jpg"))
            kp, de = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), static_mask([]))
            out.append((Rotation.from_rotvec(v["rvec"]).as_matrix(), v["f"], np.float32([q.pt for q in kp]), de))
        _REFS[key] = out
    return _REFS[key]


def ref_gauge(view, uid, every=6, verbose=False):
    """Absolute rotation + focal length of the rally frame that best matches a clicked frame of the same camera
    (same centre, so the two differ by a rotation and zoom only) -> (frame k, R, f, inliers) or None."""
    R_ = refs(view, uid.split("_")[0])
    sift = cv2.SIFT_create(2000); bf = cv2.BFMatcher()
    pp = pose_path(view, uid); kps = load_pose(pp) if pp else []
    cap = cv2.VideoCapture(clip(view, uid)); k = 0; best = None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if k % every == 0:
            kp, de = sift.detectAndCompute(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), static_mask(kps[k] if k < len(kps) else []))
            if de is not None and len(kp) >= 20:
                P = np.float32([q.pt for q in kp])
                for Rr, fr, Pr, Dr in R_:
                    m = bf.knnMatch(Dr, de, k=2)
                    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
                    if len(good) < 15:
                        continue
                    a, b = Pr[[g.queryIdx for g in good]], P[[g.trainIdx for g in good]]
                    H, mk = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
                    if mk is None or mk.sum() < 15 or (best is not None and mk.sum() <= best[3]):
                        continue
                    i = mk.ravel() > 0
                    r = least_squares(_rot_res, np.zeros(4), args=(fr, a[i], b[i]), loss="huber", f_scale=2.0, max_nfev=60)
                    rms = np.sqrt(np.median(r.fun ** 2))
                    if rms < 3.0:
                        best = (k, Rotation.from_rotvec(r.x[:3]).as_matrix() @ Rr, fr * np.exp(r.x[3]), int(mk.sum()))
        k += 1
    if verbose and best:
        print("  reference match at frame", best[0], "inliers", best[3], "f %.0f" % best[2], flush=True)
    return best


def calib_rally(view, uid, C, k1, shift, verbose=False):
    """Every frame's rotation and focal length for one rally (C fixed) -> dict of arrays."""
    K1[0] = k1
    n, pairs = load_match(view, uid)
    EV = line_evidence(view, uid)
    if not EV:
        return None
    ob = pack_obs(player_obs(view, uid, shift))
    LF = [(k, *EV[k][:2], EV[k][2]) for k in sorted(EV, key=lambda k: -EV[k][2])[:10]]
    rg = ref_gauge(view, uid, verbose=verbose)
    if rg is not None:
        a, Ra, fa, _ = rg
        Rrel, f = chain_from(pairs, a, fa)
        x = np.r_[Rotation.from_matrix(Ra).as_rotvec(), 0.0]
        x, sc = gauge_line_search(C, x, Rrel, f, LF, span=(1.0, 1.0, 0.6, 0.05), steps=(5, 5, 3, 3))
        R = np.einsum("nij,jk->nik", Rrel, Rotation.from_rotvec(x[:3]).as_matrix()); F = f * np.exp(x[3])
        return _refine_frames(R, F, C, EV, n, sc, ref=rg[3])
    a = max(EV, key=lambda k: EV[k][2])  # anchor: the frame with the most paint
    best = None
    for fa in (450, 650, 900, 1250, 1700, 2300):
        Rrel, f = chain_from(pairs, a, fa)
        x, px = fit_gauge(C, Rrel, f, ob) if len(ob[0]) else (np.r_[Rotation.from_matrix(look_at(C, np.array([3.05, 6.7, 0]))).as_rotvec(), 0.0], 99)
        x, sc = gauge_line_search(C, x, Rrel, f, LF, span=(8, 8, 3, 0.3), steps=(9, 9, 3, 5))
        if verbose:
            print("  f_anchor", fa, "score %.3f players %.0f px" % (sc, px), flush=True)
        if best is None or sc > best[0]:
            best = (sc, x, Rrel, f)
    sc, x, Rrel, f = best
    x, sc = gauge_line_search(C, x, Rrel, f, LF, span=(1.5, 1.5, 1.0, 0.08), steps=(7, 7, 5, 5))
    R = np.einsum("nij,jk->nik", Rrel, Rotation.from_rotvec(x[:3]).as_matrix()); F = f * np.exp(x[3])
    return _refine_frames(R, F, C, EV, n, sc, ref=0)


def _refine_frames(R, F, C, EV, n, sc, ref):
    # per-frame line refinement where paint is visible; corrections interpolated over time (drift, zoom error)
    ks, dv, df, q = [], [], [], {}
    for k, (dt, pts, npx) in sorted(EV.items()):
        R2, f2, on, med = fit_frame_lines(R[k], F[k], C, dt)
        p_, r_ = line_score(R2, f2, C, dt, pts)
        d = Rotation.from_matrix(R2 @ R[k].T).as_rotvec()
        if p_ >= 0.7 and r_ >= 0.3 and np.degrees(np.linalg.norm(d)) < 4 and abs(np.log(f2 / F[k])) < 0.15:
            ks.append(k); dv.append(d); df.append(np.log(f2 / F[k])); q[k] = (p_, r_)
    if ks:
        t = np.arange(n)
        D = np.stack([np.interp(t, ks, np.array(dv)[:, j]) for j in range(3)], 1)
        R = np.einsum("nij,njk->nik", Rotation.from_rotvec(D).as_matrix(), R)
        F = F * np.exp(np.interp(t, ks, df))
    gap = np.array([min([abs(k - g) for g in ks]) if ks else 999 for k in range(n)])
    return {"R": R, "f": F, "anchored": np.array(ks, int), "gap": gap, "gauge_score": sc, "ref_inliers": ref,
            "prec": np.array([q[k][0] for k in ks]), "rec": np.array([q[k][1] for k in ks])}


SHIFT = -7  # player-camera frame = TV frame + SHIFT (shuttle reprojection scan on 30 dev rallies, both cameras; = side camera)


def rally(args):
    view, uid = args
    dst = os.path.join(pdir(view, "calib"), uid + ".npz")
    if os.path.exists(dst):
        return uid, "done"
    cv2.setNumThreads(1)
    try:
        if not os.path.exists(os.path.join(pdir(view, "match"), uid + ".npz")):
            match((view, uid))
        S = json.load(open(os.path.join(ROOT, "outputs", "cctv", f"pcam_{view}_{uid.split('_')[0]}.json")))
        C = np.array(S["C"])
        r = calib_rally(view, uid, C, S["k1"], SHIFT)
        if r is None:
            return uid, "no frame shows the court"
        np.savez_compressed(dst, C=C, k1=S["k1"], shift=SHIFT, rvec=Rotation.from_matrix(r["R"]).as_rotvec(), f=r["f"],
                            anchored=r["anchored"], gap=r["gap"], prec=r["prec"], rec=r["rec"], gauge_score=r["gauge_score"])
        return uid, f"{len(r['f'])} frames, {len(r['anchored'])} line-checked (prec {r['prec'].mean() if len(r['prec']) else 0:.2f}), gap median {np.median(r['gap']):.0f}"
    except Exception as e:
        return uid, "FAILED " + repr(e)[:200]


def main():
    stage, view = sys.argv[1], sys.argv[2]
    uids = sys.argv[3:] or open(os.path.join(ROOT, "outputs", "cctv_pcam_dev_uids.txt")).read().split()
    if stage == "match":
        with Pool(int(os.environ.get("NPROC", "16"))) as pool:
            for u, msg in pool.imap_unordered(match, [(view, u) for u in uids]):
                print(u, msg, flush=True)
    elif stage == "rally":
        with Pool(int(os.environ.get("NPROC", "16"))) as pool:
            for u, msg in pool.imap_unordered(rally, [(view, u) for u in uids]):
                print(u, msg, flush=True)


if __name__ == "__main__":
    main()
