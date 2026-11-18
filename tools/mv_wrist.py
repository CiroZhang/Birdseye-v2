"""Hitter evidence from 3D wrists: each tracked player's wrists triangulated from every camera that sees them
(tools/mv_track.py person -> per-camera pose detection), measured against the shuttle's camera rays at a hit.

The shuttle at contact lies on each camera's ray through its detected pixel, so a wrist's distance to those rays needs
no shuttle depth: the hitter's racket hand is within a racket's length of them, the partner usually is not.
reach(f) -> (K,) smallest wrist-to-shuttle-ray distance per player over frames f-W..f+W (inf where unknown).
usage: python tools/mv_wrist.py <answer key json (multi-camera ids)>   (accuracy of the nearest-wrist hitter)"""
import json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import mv_people as MP  # noqa: E402

WRISTS = (9, 10)
KP_CONF = 0.3
W = 2


def closest_point(C, D):
    """least-squares point nearest to rays (C (n,3), D (n,3) unit) and the rms distance to them"""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for c, d in zip(C, D):
        M = np.eye(3) - np.outer(d, d); A += M; b += M @ c
    X = np.linalg.solve(A, b)
    r = [np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - c)) for c, d in zip(C, D)]
    return X, float(np.sqrt(np.mean(np.square(r))))


def ray_dist(X, c, d):
    return float(np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (X - c)))


class Wrists:
    def __init__(s, uid):
        s.uid = uid; s.pe = MP.People(uid)
        s.m = np.load(os.path.join(ROOT, "outputs", "cctv", "mv_track", uid + ".npz")); s.K = int(s.m["K"])
        s.shut = {}                                          # view -> {TV frame: pixel}
        import cctv_tri as CT
        for v, base, sh in (("tv", "cctv", 0), ("side", "cctv_side", (CT.side_camera(uid) or (None, 0))[1])):
            p = os.path.join(ROOT, "outputs", base, "refined", uid + ".json")
            if os.path.exists(p) and v in s.pe.views:
                r = json.load(open(p))
                s.shut[v] = {int(f) - sh: np.array(px, float) for f, px in zip(r["frames"], r["px"]) if px is not None}

    def wrists3d(s, f, k):
        """[(X (3,), rms)] for player k's wrists at TV frame f, from >= 2 cameras"""
        out = []
        for w in WRISTS:
            C, D = [], []
            for v in MP.ORDER if hasattr(MP, "ORDER") else ("tv", "side", "p1", "p2"):
                if v not in s.pe.views:
                    continue
                i = int(s.m[v + "_idx"][f, k])
                kk = s.pe.frame_of(v, f)
                if i < 0 or kk is None:
                    continue
                camf, _, Kp = s.pe.views[v]; cam = camf(kk)
                if cam is None or i >= len(Kp[kk]) or Kp[kk][i][w, 2] < KP_CONF:
                    continue
                C.append(np.asarray(cam["C"], float)); D.append(MP.rays(cam, Kp[kk][i][w, :2])[0])
            if len(C) >= 2:
                out.append(closest_point(np.array(C), np.array(D)))
        return out

    def shuttle_rays(s, f):
        """camera rays through the shuttle at TV frame f, TV first (empty when the TV does not see it)"""
        out = []
        if f not in s.shut.get("tv", {}):
            return out
        for v, tr in s.shut.items():
            if f in tr:
                kk = s.pe.frame_of(v, f); cam = s.pe.views[v][0](kk) if kk is not None else None
                if cam is not None:
                    out.append((np.asarray(cam["C"], float), MP.rays(cam, tr[f])[0]))
        return out

    def reach_all(s, f):
        """(K,2): per player the smallest wrist distance over frames f-W..f+W to the TV shuttle ray alone (tells
        partners apart: the TV looks down the court) and to all shuttle rays when >= 2 cameras see it (inf unknown)"""
        best = np.full((s.K, 2), np.inf)
        T = len(s.m["foot"])
        for t in range(max(0, f - W), min(T, f + W + 1)):
            R = s.shuttle_rays(t)
            if not R:
                continue
            for k in range(s.K):
                for X, rms in s.wrists3d(t, k):
                    if rms > 0.3:
                        continue
                    best[k, 0] = min(best[k, 0], ray_dist(X, *R[0]))
                    if len(R) >= 2:
                        best[k, 1] = min(best[k, 1], max(ray_dist(X, c, d) for c, d in R))
        return best

    def reach(s, f):
        best = np.full(s.K, np.inf)
        T = len(s.m["foot"])
        for t in range(max(0, f - W), min(T, f + W + 1)):
            R = s.shuttle_rays(t)
            if len(R) < 2:                                    # one ray spans the whole court: no position
                continue
            for k in range(s.K):
                for X, rms in s.wrists3d(t, k):
                    if rms > 0.3:                             # cameras disagree: a mismatched detection
                        continue
                    best[k] = min(best[k], max(ray_dist(X, c, d) for c, d in R))   # must be near every camera's ray
        return best


def main(key):
    GT = json.load(open(key)); ok = n = unk = 0; D = []
    for u, hs in GT.items():
        w = Wrists(u)
        for f, who in hs:
            k = int(who.split()[-1]) - 1
            if k < 0:
                continue
            r = w.reach(f); n += 1
            if not np.isfinite(r).any():
                unk += 1; continue
            ok += int(np.argmin(r) == k)
            if np.isfinite(r[k]):
                D.append(r[k])
            if np.argmin(r) != k:
                print(f"  {u} f{f} true {k} picked {int(np.argmin(r))} reach {np.round(r, 2)}")
    print(f"nearest 3D wrist to the shuttle rays: right {ok}/{n - unk} ({ok / max(n - unk, 1):.0%}), no evidence {unk}/{n}; "
          f"hitter wrist-ray distance median {np.median(D):.2f} m p90 {np.percentile(D, 90):.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
