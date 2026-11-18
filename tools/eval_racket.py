"""Back-view 3D rackets checked from the TOP drone (never used by the pipeline).

Top view: YOLOv8x racket boxes (tools/top_racket.py). Each box is given to the player whose 3D hand projects
nearest (< MAX_M); the racket tip is the box corner farthest from that hand. Our racket (VIMO hand + direction,
tools/fb_full_example.rackets, as in the viewer) is projected into the top camera and scored on
  tip    distance from the projected tip to that corner (m, at the top camera's ~1 m-height scale)
  angle  in-court direction (hand -> tip) vs the box diagonal, on boxes >= MIN_LEN long (degrees)
split by whether a back-view racket box was seen (solid) or the direction is the forearm guess (faded).
usage: python tools/eval_racket.py BD15/Rally-39 ...   (env DRONE_VIMO=drone_vimo_bc, DRONE_CALIB=calib_bc)"""
import json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from birdseye_v2.court import Camera  # noqa: E402
from birdseye_v2.players import load_rackets  # noqa: E402
from birdseye_v2.tracks import RACKET_LEN  # noqa: E402
import fb_full_example as FE  # noqa: E402

DRONE_VIMO = os.environ.get("DRONE_VIMO", "drone_vimo_bc")
DRONE_CALIB = os.environ.get("DRONE_CALIB", "calib_bc")
MAX_M, MIN_LEN = 0.9, 0.35  # m


def our_rackets(cam, VJ, boxes, fn=None):
    """-> list over players of (hand (T,3), tip (T,3), seen (T,))."""
    fn = fn or FE.rackets
    out = []
    for J in VJ:
        if np.isfinite(J[:, 0, 0]).sum() < 8:
            continue
        w, u, near, _ = fn(cam, J, boxes)
        out.append((w, w + (RACKET_LEN - 0.08) * u, near))
    return out


def score(mr, fn=None):
    m, r = mr.split("/")
    cal = json.load(open(os.path.join(ROOT, "outputs", DRONE_CALIB, m, r + ".json")))
    top, back = Camera.from_dict(cal["top"]), Camera.from_dict(cal["back"])
    off = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))["sync"]["offset"]
    VJ = np.load(os.path.join(ROOT, "outputs", DRONE_VIMO, m, r + ".npz"))["joints"]
    boxes = load_rackets(os.path.join(ROOT, "outputs", "racket", m, r + ".npz"))
    T = VJ.shape[1]
    boxes = (boxes + [np.zeros((0, 5))] * T)[:T]
    rk = our_rackets(back, VJ, boxes, fn)
    d = np.load(os.path.join(ROOT, "outputs", "racket_top", m, r + ".npz"))
    idx = np.r_[0, np.cumsum(d["counts"])]
    s = np.linalg.norm(np.diff(top.project(np.array([[3.0, 6.7, 1.0], [4.0, 6.7, 1.0]])), axis=0))  # px per m
    res = []
    for k, ft in enumerate(d["frames"]):
        fb = int(round(ft + off))
        if not 0 <= fb < T:
            continue
        B = d["boxes"][idx[k]:idx[k + 1]]
        B = B[B[:, 4] >= 0.3]
        players = [(top.project(h[fb][None])[0], top.project(t[fb][None])[0], bool(sn[fb]))
                   for h, t, sn in rk if np.isfinite(h[fb]).all() and np.isfinite(t[fb]).all()]
        if not players or not len(B):
            continue
        used = set()
        for b in B:
            corners = np.array([[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]])
            c = corners.mean(0)
            j = int(np.argmin([np.linalg.norm(c - h) for h, _, _ in players]))
            h, t, sn = players[j]
            if j in used or np.linalg.norm(c - h) / s > MAX_M:
                continue
            used.add(j)
            E = corners[np.argmax(np.linalg.norm(corners - h, axis=1))]
            row = {"tip": float(np.linalg.norm(t - E) / s), "seen": sn, "len": float(np.linalg.norm(E - h) / s)}
            if row["len"] >= MIN_LEN and np.linalg.norm(t - h) > 1:
                a, bb = E - h, t - h
                row["angle"] = float(np.degrees(np.arccos(np.clip(a @ bb / np.linalg.norm(a) / np.linalg.norm(bb), -1, 1))))
            res.append(row)
    return res


def summary(rows):
    out = {}
    for name, sel in (("all", lambda x: True), ("seen", lambda x: x["seen"]), ("guess", lambda x: not x["seen"])):
        R = [x for x in rows if sel(x)]
        if not R:
            continue
        tip = np.array([x["tip"] for x in R])
        ang = np.array([x["angle"] for x in R if "angle" in x])
        out[name] = {"n": len(R), "tip_median_m": round(float(np.median(tip)), 3), "tip_mean_m": round(float(tip.mean()), 3),
                     "n_angle": len(ang), "angle_median_deg": round(float(np.median(ang)), 1) if len(ang) else None,
                     "angle_within_30deg": round(float((ang < 30).mean()), 3) if len(ang) else None}
    return out


if __name__ == "__main__":
    rows = [x for mr in sys.argv[1:] for x in score(mr)]
    print(json.dumps(summary(rows), indent=1))
