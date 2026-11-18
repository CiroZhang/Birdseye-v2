"""Two-view 3D shuttle for the CCTV rallies: the TV view (per-clip BirdsCourt camera, outputs/cctv/calib_bc) and the
fixed side camera (outputs/cctv/side_cam_<session>.json, tools/cctv_sidecal.py, once per session), both shuttle tracks refined
(outputs/cctv{,_side}/refined). The clips are cut synced, so frame k is the same instant in both.

The court is symmetric under a 180 deg turn about its centre, so the side camera's world frame may be that turn away
from the TV camera's, and the side feed's video can lag its audio (the audio sync) by a few frames; the session takes the
turn and frame shift that make the two views' rays meet (median gap over a sample of rallies).
A frame is kept when the rays pass within MAX_GAP of each other and the midpoint is in the hall above the court.

Output: outputs/cctv/tri/<uid>.json {shuttle: [[frame, x, y, z], ...], gap_med, n_tv, n_side}  (DroneData _gt.json style)
usage: python tools/cctv_tri.py [uid ...]"""
import glob, json, os, sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import L, W, Camera  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "cctv", "tri")
MAX_GAP = 0.1  # m (well-matched frames meet within ~2 cm)
SHIFTS = range(-15, 16)  # side frame = TV frame + shift


def turn180(cam):
    A, b = np.diag([-1.0, -1.0, 1.0]), np.array([W, L, 0.0])  # X_old = A X_new + b
    return Camera(cam.f, cv2.Rodrigues(cam.R @ A)[0].ravel(), cam.R @ b + cam.tvec, cam.size, cam.dist)


def track(path):
    if not os.path.exists(path):
        return {}
    t = json.load(open(path))
    return dict(zip(t["frames"], np.array(t["px"], float).reshape(-1, 2)))


def closest(c1, d1, c2, d2):
    """Midpoint and gap of the closest approach of rays c + s d (rows)."""
    w = c1 - c2
    a, b, c = (d1 * d1).sum(1), (d1 * d2).sum(1), (d2 * d2).sum(1)
    d, e = (d1 * w).sum(1), (d2 * w).sum(1)
    den = np.maximum(a * c - b * b, 1e-9)
    s, t = (b * e - c * d) / den, (a * e - b * d) / den
    p1, p2 = c1 + s[:, None] * d1, c2 + t[:, None] * d2
    return (p1 + p2) / 2, np.linalg.norm(p1 - p2, axis=1), s, t


def tri(uid, side, shift=0):
    cp = os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")
    c = json.load(open(cp)).get("cam") if os.path.exists(cp) else None
    if not c:
        return None
    tv = Camera.from_dict(c)
    a, b = track(os.path.join(ROOT, "outputs", "cctv", "refined", uid + ".json")), \
        track(os.path.join(ROOT, "outputs", "cctv_side", "refined", uid + ".json"))
    fr = sorted(k for k in a if k + shift in b)
    if not fr:
        return {"frames": np.zeros(0, int), "X": np.zeros((0, 3)), "gap": np.zeros(0), "n_tv": len(a), "n_side": len(b)}
    c1, d1 = tv.rays(np.array([a[k] for k in fr]))
    c2, d2 = side.rays(np.array([b[k + shift] for k in fr]))
    X, gap, s, t = closest(np.broadcast_to(c1, d1.shape), d1, np.broadcast_to(c2, d2.shape), d2)
    ok = (s > 0) & (t > 0)
    return {"frames": np.array(fr)[ok], "X": X[ok], "gap": gap[ok], "n_tv": len(a), "n_side": len(b)}


def session_of(uid):
    return uid.rsplit("_", 1)[0]


def side_cam_raw(sess):
    """The session's side camera as calibrated (tools/cctv_sidecal.py; the first session's file has no suffix)."""
    p = os.path.join(ROOT, "outputs", "cctv", f"side_cam_{sess}.json")
    p = p if os.path.exists(p) else os.path.join(ROOT, "outputs", "cctv", "side_cam.json")
    return Camera.from_dict(json.load(open(p))["cam"])


def side_camera(uid):
    """(side camera in the TV camera's world frame, frame shift) for a triangulated rally, or None."""
    tp = os.path.join(OUT, uid + ".json")
    if not os.path.exists(tp):
        return None
    t = json.load(open(tp))
    cam = side_cam_raw(session_of(uid))
    return (turn180(cam) if t.get("turned", True) else cam), t.get("shift", 0)


def main():
    uids = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(ROOT, "outputs", "cctv_side", "refined", "*.json")))
    os.makedirs(OUT, exist_ok=True)
    for sess in sorted(set(map(session_of, uids))):
        su = [u for u in uids if session_of(u) == sess]
        side0 = side_cam_raw(sess)
        probe = su[:: max(1, len(su) // 60)]
        med = {}
        for name, cam in (("as is", side0), ("turned", turn180(side0))):
            for sh in SHIFTS:
                g = [r["gap"] for r in (tri(u, cam, sh) for u in probe) if r is not None and len(r["gap"])]
                med[name, sh] = float(np.median(np.concatenate(g))) if g else np.inf
        (name, shift), best = min(med.items(), key=lambda kv: kv[1])
        side = side0 if name == "as is" else turn180(side0)
        print(f"{sess} side camera: {name}, shift {shift} frames, probe median gap {best:.3f} m "
              f"(unshifted: as is {med['as is', 0]:.3f}, turned {med['turned', 0]:.3f})", flush=True)
        kept, both, rallies = 0, 0, 0
        for u in su:
            r = tri(u, side, shift)
            if r is None:
                continue
            X, gap = r["X"], r["gap"]
            ok = (gap < MAX_GAP) & (X[:, 2] > -0.1) & (X[:, 2] < 15) & (X[:, 0] > -3) & (X[:, 0] < W + 3) &                  (X[:, 1] > -4) & (X[:, 1] < L + 4)
            rows = [[int(f)] + [round(float(v), 4) for v in x] for f, x in zip(r["frames"][ok], X[ok])]
            json.dump({"shuttle": rows, "gap_med": float(np.median(gap)) if len(gap) else None,
                       "n_tv": r["n_tv"], "n_side": r["n_side"], "shift": shift, "turned": name == "turned"}, open(os.path.join(OUT, u + ".json"), "w"))
            kept, both, rallies = kept + len(rows), both + len(gap), rallies + 1
        print(f"{sess}: {rallies} rallies: {both} frames seen by both views, {kept} kept (gap < {MAX_GAP} m)", flush=True)


if __name__ == "__main__":
    main()
