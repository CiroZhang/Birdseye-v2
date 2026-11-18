"""Multi-camera player bodies: VIMO (TRAM) poses placed where every calibrated camera sees them.

VIMO lifts each player from the TV view alone, so its court position is only as good as one camera's depth (0.4 m
median, 1.3 m p90 off what the player cameras see). Here the pose (24 SMPL joints relative to the pelvis) is kept and
the pelvis track r_k of each player is fitted to the 2D keypoints (YOLOv8x-pose) of the TV, side and both player
cameras: robust reprojection residuals of 12 body joints, a weak prior to VIMO's own placement and a smooth-motion
term (pelvis acceleration). Detections are matched to bodies per camera frame (Hungarian on projected distance),
twice: with VIMO's placement, then with the fit. Player-camera frames are on their fractional TV time.

Output outputs/cctv/<BMV_DIR or body_mv>/<uid>.npz, same keys as vimo4 (frames, bottom0..top1 (T,24,3)) plus
<slot>_nobs (T,4) matched detections per camera (tv, side, p1, p2).
Env: BMV_DROP=p2 (comma list) leaves cameras out (held-out check: python tools/body_mv.py --eval uid ...).
usage: python tools/body_mv.py uid ...   |   python tools/body_mv.py --eval uid ..."""
import json, os, sys

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))

COCO = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]   # shoulders, elbows, wrists, hips, knees, ankles
SMPL = [16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
SLOTS = ("bottom0", "bottom1", "top0", "top1")
VIEWS = ("tv", "side", "p1", "p2")
SIG_PX = {"tv": 4.0, "side": 4.0, "p1": 12.0, "p2": 12.0}  # keypoint noise + calibration error per camera
KP_CONF = 0.5
SIG_PRIOR = np.array([1.0, 1.0, 2.0])  # m: VIMO's placement as a weak prior (x, y; its height is not trusted)
SIG_GROUND = float(os.environ.get("BMV_SIG_GROUND", "0.15"))  # m: lower foot above the floor (jumps stay possible where the cameras say so)
SIG_BELOW = 0.01                        # m: a foot below the floor
FOOT_Z = 0.04                           # m: SMPL foot joint height of a foot on the floor
MAX_FILL = 50                           # frames: gaps in VIMO's track bridged (pose interpolated, placed by the cameras)
SIG_ACC = 0.05                          # m/frame^2: pelvis acceleration (~30 m/s^2 at 25 fps)
OUT = os.path.join(ROOT, "outputs", "cctv", os.environ.get("BMV_DIR", "body_mv"))
DROP = set(filter(None, os.environ.get("BMV_DROP", "").split(",")))


def load_kp(p):
    z = np.load(p); c = np.r_[0, np.cumsum(z["counts"])]
    return [z["kp"][c[i]:c[i + 1]] for i in range(len(z["counts"]))]


def static_cam(cam):
    R = cam.R
    return dict(R=R, C=cam.center, f=cam.f, cx=cam.size[0] / 2, cy=cam.size[1] / 2, k1=cam.dist[0], k2=cam.dist[1])


def project(cam, X):
    """X (...,3) -> (uv (...,2), depth (...)) for a camera dict (pinhole + 2-term radial)."""
    Y = (X - cam["C"]) @ cam["R"].T
    z = Y[..., 2]
    x, y = Y[..., 0] / z, Y[..., 1] / z
    r2 = x * x + y * y
    d = 1 + cam["k1"] * r2 + cam["k2"] * r2 * r2
    return np.stack([cam["f"] * x * d + cam["cx"], cam["f"] * y * d + cam["cy"]], -1), z


def cameras(uid):
    """[(view, frame->camera dict, frame->TV time, keypoints per frame)]"""
    import cctv_teach as CTE
    import cctv_tri as CT
    import pcam_calib as PC
    from birdseye_v2.court import Camera
    tv = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")))["cam"])
    side, sh = CT.side_camera(uid)
    out = []
    for v, cam, base, shift in (("tv", tv, "cctv", 0), ("side", side, "cctv_side", sh)):
        p = os.path.join(ROOT, "outputs", base, "pose1280", uid + ".npz")
        if os.path.exists(p) and v not in DROP:
            c = static_cam(cam)
            out.append((v, lambda k, c=c: c, lambda k, s=shift: k - s, load_kp(p)))
    for v in ("p1", "p2"):
        cp = os.path.join(ROOT, "outputs", "cctv_" + v, "pcam", "calib", uid + ".npz")
        pp = os.path.join(ROOT, "outputs", "cctv_" + v, "pose1280", uid + ".npz")
        if not os.path.exists(pp):                       # player cameras' standard pose (all rallies)
            pp = os.path.join(ROOT, "outputs", "cctv_" + v, "pose", uid + ".npz")
        if v in DROP or not (os.path.exists(cp) and os.path.exists(pp)):
            continue
        z = np.load(cp)
        R = Rotation.from_rotvec(z["rvec"]).as_matrix()
        cams = [dict(R=R[k], C=z["C"], f=z["f"][k], cx=PC.CX, cy=PC.CY, k1=float(z["k1"]), k2=0.0) for k in range(len(R))]
        out.append((v, lambda k, c=cams: c[k] if k < len(c) else None, lambda k, s=CTE.PCAM_SYNC[v]: k - s, load_kp(pp)))
    return out


def gather(cams, bodies, vfr):
    """Match every camera frame's detections to the bodies (current pelvis tracks) -> observation arrays."""
    pos = {f: i for i, f in enumerate(vfr)}
    obs = []
    for vi, (v, camf, tvt, K) in enumerate(cams):
        for k, D in enumerate(K):
            if len(D) == 0:
                continue
            cam = camf(k)
            if cam is None:
                continue
            t = tvt(k); f0 = int(np.floor(t)); a = t - f0
            if f0 not in pos or f0 + 1 not in pos:
                if a < 1e-6 and f0 in pos:
                    i0 = i1 = pos[f0]
                else:
                    continue
            else:
                i0, i1 = pos[f0], pos[f0 + 1]
            B, P = [], []
            for b, (L, r) in enumerate(bodies):
                if not (np.isfinite(L[i0]).all() and np.isfinite(L[i1]).all()):
                    continue
                X = (1 - a) * (L[i0] + r[i0]) + a * (L[i1] + r[i1])
                uv, dep = project(cam, X[SMPL])
                if (dep > 0.3).all():
                    B.append(b); P.append(uv)
            if not B:
                continue
            Dk = D[:, COCO]
            cost = np.full((len(B), len(D)), 1e9)
            for bi, uv in enumerate(P):
                size = np.ptp(uv[:, 1]) + 1  # projected body height in px
                for d in range(len(D)):
                    m = Dk[d, :, 2] > KP_CONF
                    if m.sum() >= 6:
                        e = np.median(np.linalg.norm(uv[m] - Dk[d, m, :2], axis=1))
                        if e < max(40.0, 0.6 * size):
                            cost[bi, d] = e
            for bi, d in zip(*linear_sum_assignment(cost)):
                if cost[bi, d] < 1e9:
                    obs.append((B[bi], vi, i0, i1, a, cam, Dk[d]))
    return obs


def bridge(Jw):
    """Bridge gaps of up to MAX_FILL frames inside a VIMO track by linear interpolation -> (joints, filled mask)."""
    ok = np.isfinite(Jw).all((1, 2))
    fill = np.zeros(len(Jw), bool)
    if ok.sum() < 2:
        return Jw, fill
    Jw = Jw.copy(); f = np.flatnonzero(ok)
    for a, b in zip(f[:-1], f[1:]):
        if 1 < b - a <= MAX_FILL + 1:
            t = (np.arange(a + 1, b) - a) / (b - a)
            Jw[a + 1:b] = (1 - t)[:, None, None] * Jw[a] + t[:, None, None] * Jw[b]
            fill[a + 1:b] = True
    return Jw, fill


def fit_body(L, raw_root, obs_b, sig, filled=None):
    """Pelvis track (T,3) for one body from its observations; filled frames (bridged gaps) have no VIMO prior."""
    T = len(L)
    ok = np.isfinite(raw_root).all(1)
    idx = np.flatnonzero(ok)
    var = {f: j for j, f in enumerate(idx)}
    n = len(idx)
    # observation rows: one per (obs, keypoint)
    rows = []
    for (vi, i0, i1, a, cam, D) in obs_b:
        m = D[:, 2] > KP_CONF
        loc = (1 - a) * L[i0][SMPL] + a * L[i1][SMPL]
        for j in np.flatnonzero(m):
            rows.append((var[i0], var[i1], a, loc[j], D[j, :2], sig[vi], cam))
    nr = len(rows)
    acc = [(var[f - 1], var[f], var[f + 1]) for f in idx if f - 1 in var and f + 1 in var]
    nres = 2 * nr + 3 * n + n + 3 * len(acc)
    J = lil_matrix((nres, 3 * n), dtype=np.int8)
    for q, (j0, j1, *_r) in enumerate(rows):
        for jj in (j0, j1):
            J[2 * q:2 * q + 2, 3 * jj:3 * jj + 3] = 1
    o = 2 * nr
    for j in range(n):
        J[o + 3 * j:o + 3 * j + 3, 3 * j:3 * j + 3] = 1
        J[o + 3 * n + j, 3 * j + 2] = 1
    o += 4 * n
    for q, trip in enumerate(acc):
        for jj in trip:
            for c in range(3):
                J[o + 3 * q + c, 3 * jj + c] = 1
    # group rows by camera dict for vectorised projection
    groups = {}
    for q, rw in enumerate(rows):
        groups.setdefault(id(rw[6]), (rw[6], []))[1].append(q)
    G = []
    for cam, qs in groups.values():
        qs = np.array(qs)
        G.append((cam, qs, np.array([rows[q][0] for q in qs]), np.array([rows[q][1] for q in qs]),
                  np.array([rows[q][2] for q in qs]), np.array([rows[q][3] for q in qs]),
                  np.array([rows[q][4] for q in qs]), np.array([rows[q][5] for q in qs])))
    acc = np.array(acc, int).reshape(-1, 3)
    r0 = raw_root[idx]
    sp = np.tile(SIG_PRIOR, (n, 1))
    if filled is not None:
        sp[filled[idx]] = 50.0
    foot = np.nanmin(L[idx][:, [10, 11], 2], axis=1)  # lower foot relative to the pelvis

    def res(p):
        r = p.reshape(n, 3)
        e = np.zeros((nr, 2))
        for cam, qs, j0, j1, a, loc, uv, s in G:
            X = loc + (1 - a)[:, None] * r[j0] + a[:, None] * r[j1]
            u, _ = project(cam, X)
            e[qs] = (u - uv) / s[:, None]
        h = r[:, 2] + foot - FOOT_Z
        out = [e.ravel(), ((r - r0) / sp).ravel(), np.where(h < 0, h / SIG_BELOW, h / SIG_GROUND)]
        if len(acc):
            out.append(((r[acc[:, 0]] - 2 * r[acc[:, 1]] + r[acc[:, 2]]) / SIG_ACC).ravel())
        return np.concatenate(out)

    # smoothed VIMO track as the start
    sol = least_squares(res, r0.ravel(), jac_sparsity=J, loss="soft_l1", f_scale=2.0, x_scale=0.1, max_nfev=60)
    out = np.full((T, 3), np.nan)
    out[idx] = sol.x.reshape(n, 3)
    return out


def run(uid):
    vb = np.load(os.path.join(ROOT, "outputs", "cctv", os.environ.get("BMV_VIMO", "vimo4"), uid + ".npz"))
    vfr = vb["frames"].astype(int)
    cams = cameras(uid)
    sig = [SIG_PX[c[0]] for c in cams]
    bodies, names, fills = [], [], []
    for s in SLOTS:
        Jw = vb[s].astype(float)
        if np.isfinite(Jw[:, 0, 0]).sum() < 10:
            continue
        Jw, fill = bridge(Jw)
        root = Jw[:, 0].copy()
        bodies.append([Jw - root[:, None], root]); names.append(s); fills.append(fill)
    # first the TV view alone with the feet on the floor (VIMO's own depth can put a far player in the air near the
    # net, where the side and player cameras would then find the umpire); then every camera, matched to that, twice
    for it in range(3):
        obs = gather(cams, bodies, vfr)
        if it == 0:
            obs = [o for o in obs if cams[o[1]][0] == "tv"]
        for b in range(len(bodies)):
            ob = [o[1:] for o in obs if o[0] == b]
            L, _ = bodies[b]
            raw = bridge(vb[names[b]].astype(float))[0][:, 0]
            bodies[b][1] = fit_body(L, raw, ob, sig, fills[b]) if ob else raw
    res = {"frames": vb["frames"]}
    for s in SLOTS:
        res[s] = np.full_like(vb[s], np.nan)
    obs = gather(cams, bodies, vfr)
    for b, s in enumerate(names):
        L, r = bodies[b]
        res[s] = (L + r[:, None]).astype(np.float32)
        no = np.zeros((len(vfr), 4), np.int16)
        for o in obs:
            if o[0] == b:
                no[o[2], VIEWS.index(cams[o[1]][0])] += 1
        res[s + "_nobs"] = no
    return res, cams


def heldout(uid, view):
    """Median px error of the bodies in a camera left out of the fit: VIMO vs the multi-camera fit."""
    global DROP
    DROP = {view}
    fit, _ = run(uid)
    DROP = set()
    vb = np.load(os.path.join(ROOT, "outputs", "cctv", os.environ.get("BMV_VIMO", "vimo4"), uid + ".npz"))
    cams = [c for c in cameras(uid) if c[0] == view]
    vfr = vb["frames"].astype(int)
    out = {}
    for name, src in (("vimo", vb), ("fit", fit)):
        bodies = []
        for s in SLOTS:
            Jw = src[s].astype(float)
            if np.isfinite(Jw[:, 0, 0]).sum() >= 10:
                bodies.append((Jw - Jw[:, :1], Jw[:, 0]))
        e = []
        for (b, vi, i0, i1, a, cam, D) in gather(cams, bodies, vfr):
            L, r = bodies[b]
            X = (1 - a) * (L[i0] + r[i0]) + a * (L[i1] + r[i1])
            uv, _ = project(cam, X[SMPL]); m = D[:, 2] > KP_CONF
            e.append(np.median(np.linalg.norm(uv[m] - D[m, :2], axis=1)))
        out[name] = (len(e), np.median(e) if e else np.nan, np.percentile(e, 90) if e else np.nan)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--eval":
        for uid in args[1:]:
            for v in ("p1", "p2", "side"):
                r = heldout(uid, v)
                print(f"{uid} held-out {v:4s}: VIMO n={r['vimo'][0]} med {r['vimo'][1]:.1f} p90 {r['vimo'][2]:.1f} px | "
                      f"fit n={r['fit'][0]} med {r['fit'][1]:.1f} p90 {r['fit'][2]:.1f} px", flush=True)
        sys.exit()
    os.makedirs(OUT, exist_ok=True)
    for uid in args:
        res, cams = run(uid)
        np.savez_compressed(os.path.join(OUT, uid + ".npz"), **res)
        tot = sum(res[s + "_nobs"].sum(0) for s in SLOTS if s + "_nobs" in res)
        print(uid, "cameras", [c[0] for c in cams], "matched", dict(zip(VIEWS, tot.tolist())), flush=True)
