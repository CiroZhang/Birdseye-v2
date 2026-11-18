"""Teacher stage for the rackets: each player's racket head in 3D, per TV frame.

Evidence (no training, all frozen detectors):
  TV racket boxes (YOLOv8x COCO 'tennis racket', outputs/cctv/racket), player-camera racket boxes (same detector on the
  close-ups, outputs/cctv_<p1|p2>/racket), players' wrists (TV pose lifted at the feet's depth, as the viewer bodies).
A box is given to the player whose wrist it is nearest to (TV: in the image; player camera: wrist to box ray, 3D).
3D: TV ray x player-camera ray where both see the racket (closest approach, gap < PAIR_GAP); one ray alone -> the point
on it nearest that player's wrist. The head stays within REACH_R of the wrist; short gaps are interpolated.
At the teacher's hits the hitter's racket head is the shuttle (the teacher's 3D at the hit frame), blended over +-3 frames.
-> per player: frames -> [x, y, z, source] (source 2 = TV + player camera, 1 = player camera, 0 = TV, 3 = hit).
usage (library): racket3d(uid, teacher_out, V) ; python tools/racket3d.py uid (prints coverage)"""
import os, sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
PAIR_GAP = 0.35   # m
REACH_R = 0.9     # m: racket head to wrist
GAP_FILL = 3      # frames
WR = (9, 10)      # COCO wrists


def lift(o):
    """(T,4,17,3) keypoints on their TV rays at the player's depth (vertical plane through the feet, facing the TV)."""
    C, D, cf, F = o["C"], o["kp_dir"], o["kp_conf"], o["foot"]
    out = np.full(D.shape, np.nan)
    for t in range(len(C)):
        for p in range(D.shape[1]):
            if not np.isfinite(F[t, p]).all():
                continue
            f = np.r_[F[t, p], 0.0]
            n = f - C[t]; n[2] = 0; n /= np.linalg.norm(n) + 1e-9
            den = D[t, p] @ n
            ok = (np.abs(den) > 1e-6) & (cf[t, p] > 0.3)
            s = np.where(ok, ((f - C[t]) @ n) / np.where(ok, den, 1), np.nan)
            out[t, p] = C[t] + s[:, None] * D[t, p]
    return out


def _boxes(path):
    if not os.path.exists(path):
        return None
    z = np.load(path)
    c = z["counts"]; b = z["boxes"]
    idx = np.r_[0, np.cumsum(c)]
    return [b[idx[i]:idx[i + 1]] for i in range(len(c))]


def _closest(c1, d1, c2, d2):
    w = c1 - c2; a, b, c = d1 @ d1, d1 @ d2, d2 @ d2; d, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-9:
        return None, 9e9
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    p1, p2 = c1 + s * d1, c2 + t * d2
    return (p1 + p2) / 2 if s > 0 and t > 0 else None, np.linalg.norm(p1 - p2)


def _near_on_ray(c, d, X):
    return c + max((X - c) @ d, 0.5) * d


def racket3d(uid, out, V):
    import cctv_teach as CTE
    from birdseye_v2.teacher_dense import ObsCams
    o = np.load(os.path.join(ROOT, "outputs", "rally_obs", "cctv", uid + ".npz"))
    K = lift(o)
    T, NP = K.shape[0], K.shape[1]
    W = K[:, :, list(WR)]                     # (T,NP,2,3)
    tv = V.cams[0]
    tvb = _boxes(os.path.join(ROOT, "outputs", "cctv", "racket", uid + ".npz")) or []
    rays_tv = {}                               # (t, p) -> (c, d)
    for t in range(min(T, len(tvb))):
        bx = tvb[t]
        if not len(bx):
            continue
        cen = np.c_[(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2]
        half = np.hypot(bx[:, 2] - bx[:, 0], bx[:, 3] - bx[:, 1]) / 2
        used = set()
        for p in range(NP):
            wp = [tv.project(w[None])[0] for w in W[t, p] if np.isfinite(w).all()]
            if not wp:
                continue
            dd = np.min([np.linalg.norm(cen - q, axis=1) for q in wp], axis=0)
            ok = np.nonzero(dd < np.maximum(2.5 * half, 25))[0]
            ok = [i for i in ok if i not in used]
            if ok:
                i = min(ok, key=lambda i: dd[i] - 40 * bx[i, 4]); used.add(i)
                c, d = tv.rays(cen[i][None]); rays_tv[(t, p)] = (np.asarray(c, float).reshape(-1)[:3], d[0])
    rays_pc = {}                               # (t, p) -> (c, d)
    for v in ("p1", "p2"):
        cp = os.path.join(ROOT, "outputs", "cctv_" + v, "pcam", "calib", uid + ".npz")
        pb = _boxes(os.path.join(ROOT, "outputs", "cctv_" + v, "racket", uid + ".npz"))
        if pb is None or not os.path.exists(cp):
            continue
        z = np.load(cp); R = Rotation.from_rotvec(z["rvec"]).as_matrix(); sh = CTE.PCAM_SYNC[v]
        for kk in range(min(len(pb), len(R))):
            bx = pb[kk]
            t = int(round(kk - sh))
            if not len(bx) or not 0 <= t < T:
                continue
            cen = np.c_[(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2]
            cam = ObsCams(np.repeat(R[kk][None], len(cen), 0), np.full(len(cen), z["f"][kk]), z["C"], float(z["k1"]))
            c, ds = cam.rays(cen)
            for i, d in enumerate(ds):
                best, bp = 1.0, None
                for p in range(NP):
                    for w in W[t, p]:
                        if np.isfinite(w).all():
                            e = np.linalg.norm(np.cross(w - c, d))
                            if e < best:
                                best, bp = e, p
                if bp is not None and ((t, bp) not in rays_pc or bx[i, 4] > rays_pc[(t, bp)][2]):
                    rays_pc[(t, bp)] = (np.asarray(c, float), d, bx[i, 4])
    H = np.full((T, NP, 3), np.nan); src = np.full((T, NP), -1)
    for t in range(T):
        for p in range(NP):
            ws = [w for w in W[t, p] if np.isfinite(w).all()]
            if not ws:
                continue
            a, b = rays_tv.get((t, p)), rays_pc.get((t, p))
            X, s = None, -1
            if a and b:
                X, g = _closest(a[0], a[1], b[0], b[1])
                s = 2 if X is not None and g < PAIR_GAP else -1
                if s < 0:
                    X = None
            if X is None and b:
                w = min(ws, key=lambda w: np.linalg.norm(np.cross(w - b[0], b[1])))
                X, s = _near_on_ray(b[0], b[1], w), 1
            if X is None and a:
                w = min(ws, key=lambda w: np.linalg.norm(np.cross(w - a[0], a[1])))
                X, s = _near_on_ray(a[0], a[1], w), 0
            if X is None:
                continue
            w = min(ws, key=lambda w: np.linalg.norm(X - w))
            r = X - w; n = np.linalg.norm(r)
            if n > REACH_R:
                X = w + r / n * REACH_R
            H[t, p], src[t, p] = X, s
    # the hitter's racket head is the shuttle at the teacher's hits
    fr = np.asarray(out["frames"], float); P = np.asarray(out["P"])
    for h in out["hits"]:
        ti = int(round(h))
        if not 0 <= ti < T:
            continue
        S = P[np.argmin(np.abs(fr - h))]
        dd = [min([np.linalg.norm(S - w) for w in W[ti, p] if np.isfinite(w).all()] or [99]) for p in range(NP)]
        p = int(np.argmin(dd))
        if dd[p] > 1.6:
            continue
        for dt in range(-3, 4):
            t = ti + dt
            if not 0 <= t < T:
                continue
            a = 1 - abs(dt) / 4
            S_t = P[np.argmin(np.abs(fr - t))]
            H[t, p] = S_t if not np.isfinite(H[t, p]).all() or dt == 0 else a * S_t + (1 - a) * H[t, p]
            src[t, p] = 3 if dt == 0 else max(src[t, p], 0)
    # short gaps
    for p in range(NP):
        ok = np.isfinite(H[:, p, 0])
        idx = np.nonzero(ok)[0]
        for i0, i1 in zip(idx[:-1], idx[1:]):
            if 1 < i1 - i0 <= GAP_FILL + 1:
                for t in range(i0 + 1, i1):
                    a = (t - i0) / (i1 - i0); H[t, p] = (1 - a) * H[i0, p] + a * H[i1, p]; src[t, p] = 0
    return H, src, K


if __name__ == "__main__":
    import json
    import cctv_teach as CTE
    uid = sys.argv[1]
    V, tri, _ = CTE.load(uid)
    out = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else json.load(open(os.path.join(ROOT, "outputs", "cctv", "teach", uid + ".json")))
    H, src, K = racket3d(uid, out, V)
    have = np.isfinite(K[..., 0, 0])
    for p in range(H.shape[1]):
        if have[:, p].sum() < 10:
            continue
        s = src[have[:, p], p]
        print(f"player {p}: racket in {np.mean(s >= 0):.0%} of frames  (TV+pcam {np.mean(s == 2):.0%}, pcam {np.mean(s == 1):.0%}, TV {np.mean(s == 0):.0%}, hit {np.mean(s == 3):.0%})")
