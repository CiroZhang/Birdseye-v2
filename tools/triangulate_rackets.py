"""Real 3D rackets from DroneData's two drones (training matches only): the back view's racket boxes (rally_obs rays,
YOLOv8x) and the top view's (tools/top_racket.py) are paired by how closely their rays meet; the meeting point is the
racket's box centre in 3D. Each is given to the nearest triangulated wrist (outputs/pose3d) within reach and
expressed in that forearm's frame exactly as RallyPhys parametrises the racket (a along the forearm, n the elbow-plane
normal, c = n x a), so the learned grip prior (birdseye_v2/gripprior.py) is a distribution over the same numbers.

Output: outputs/racket3d.npz {local (N,3) unit direction wrist -> box centre in the forearm frame, dist (N,),
  hand (N,) 0 left / 1 right, uid (N,), t (N,) frame, X (N,3) box centre}
usage: python tools/triangulate_rackets.py   (env DRONE_TRAIN=BD26+..., uses BD12-14 + DRONE_TRAIN; TEST=1: the scored
  matches -> outputs/racket3d_test.npz, for evaluation only)"""
import glob, json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ.setdefault("VTEACH", "0")
import train_rallynet as TR  # noqa: E402

MEET = 0.12  # m: back and top rays must pass this close
REACH = (0.08, 0.55)  # m: box centre to wrist


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def forearm_frame(J, hand):
    """RallyPhys's racket frame for one hand (birdseye_v2/rallyphys.py body()): a, n, c."""
    el, wr, sh = (J[7], J[9], J[5]) if hand == 0 else (J[8], J[10], J[6])
    neck = (J[5] + J[6]) / 2
    a = unit(wr - el)
    ref = unit(sh - el) + 0.3 * unit(J[5] - neck)
    n = unit(np.cross(a, ref))
    return a, n, np.cross(n, a)


def closest(c1, d1, c2, d2):
    """Midpoint and distance of the closest approach of two rays."""
    w = c1 - c2
    b = d1 @ d2
    den = 1 - b * b
    if den < 1e-6:
        return None, 1e9
    s = (b * (d2 @ w) - (d1 @ w)) / den
    t = ((d2 @ w) - b * (d1 @ w)) / den
    if s <= 0 or t <= 0:
        return None, 1e9
    p, q = c1 + s * d1, c2 + t * d2
    return (p + q) / 2, np.linalg.norm(p - q)


def run():
    L, D, H, U, TT, XX = [], [], [], [], [], []
    test = os.environ.get("TEST") == "1"  # the scored matches instead (racket evaluation only, never trained on)
    paths = [p for p in sorted(glob.glob(os.path.join(TR.OBS, "drone", "*.npz")))
             if os.path.basename(p).startswith(TR.TRAIN_DRONE) != test]
    for p in paths:
        uid = os.path.basename(p)[:-4]
        m, r = uid.split("__")
        pp = os.path.join(ROOT, "outputs", "pose3d", m, r + ".npz")
        if not os.path.exists(pp) and not test:
            continue
        o = np.load(p)
        T = len(o["C"])
        Ct, Dt, Mt = TR.top_rackets(uid, T)
        byf = {}
        if os.path.exists(pp):
            pz = np.load(pp)
            for fr, J in zip(pz["frames"], pz["J"]):
                byf.setdefault(int(fr), []).append(J)
        for t in range(T):
            if Mt[t].sum() == 0 or (t not in byf and not test) or not np.isfinite(o["C"][t]).all():
                continue
            for k in range(o["rk_dir"].shape[1]):
                if o["rk_conf"][t, k] <= 0.25:
                    continue
                db = unit(o["rk_dir"][t, k])
                best = None
                for j in np.where(Mt[t] > 0)[0]:
                    X, e = closest(o["C"][t].astype(float), db, Ct[t].astype(float), Dt[t, j].astype(float))
                    if X is not None and e < MEET and (best is None or e < best[1]):
                        best = (X, e)
                if best is None or not (0.0 < best[0][2] < 3.2):
                    continue
                X = best[0]
                if test:  # evaluation: the 3D box centre alone (scored against the model's own hands)
                    L.append([np.nan] * 3); D.append(np.nan); H.append(-1); U.append(uid); TT.append(t); XX.append(X)
                    continue
                cand = []
                for J in byf[t]:
                    for hand in (0, 1):
                        w, e_ = J[9 + hand], J[7 + hand]
                        if np.isfinite(w).all() and np.isfinite(e_).all() and np.isfinite(J[[5, 6]]).all():
                            cand.append((np.linalg.norm(X - w), J, hand))
                if not cand:
                    continue
                cand.sort(key=lambda c: c[0])
                dist, J, hand = cand[0]
                if not REACH[0] < dist < REACH[1] or (len(cand) > 1 and cand[1][0] < dist + 0.1):
                    continue  # unclear whose hand (or which hand) holds it
                a, n, c = forearm_frame(J, hand)
                d = unit(X - J[9 + hand])
                L.append([d @ a, d @ n, d @ c]); D.append(dist); H.append(hand); U.append(uid); TT.append(t); XX.append(X)
        print(uid, len(L), flush=True)
    np.savez(os.path.join(ROOT, "outputs", "racket3d_test.npz" if test else "racket3d.npz"), local=np.array(L, np.float32),
             dist=np.array(D, np.float32), hand=np.array(H), uid=np.array(U), t=np.array(TT), X=np.array(XX, np.float32))
    L = np.array(L)
    print("rackets", len(L), "dist median %.2f" % np.nanmedian(D), "angle to forearm deg p25/50/75",
          np.nanpercentile(np.degrees(np.arccos(np.clip(L[:, 0], -1, 1))), [25, 50, 75]).round(0))


if __name__ == "__main__":
    run()
