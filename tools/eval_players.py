"""Player position error of the back-view pipeline against DroneData's top-view player annotations.

GT: CSV player{1..4}x/y (top-view px, top clock) -> floor via the calibrated top camera.
Prediction (back view only, back clock = top clock + sync offset):
  raw      per-frame YOLOv8x-pose mid-ankle ray-cast to the floor (birdseye_v2.players.foot_point)
  tracked  the viewer's tracked, gap-filled, smoothed feet (birdseye_v2.tracks)
Per frame, GT and predicted players are matched one-to-one (Hungarian); an unmatched GT player
counts as missed. Same matching idea as Birdseye v1's compute_doubles_final_error.py (0.8 m tolerance).
usage: python tools/eval_players.py BD12/Rally-1 ... | BD15 BD16 ...   -> outputs/eval/players_<tag>.json"""
import csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import linear_sum_assignment

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import Camera
from birdseye_v2.footlock import stabilize
from birdseye_v2.players import load_pose, players_3d
from birdseye_v2.tracks import smooth_track, track_players

DRONE_CALIB = os.environ.get("DRONE_CALIB", "calib")  # back camera; the top camera (GT) is the same in both
DRONE_VIMO = os.environ.get("DRONE_VIMO", "drone_vimo")
MAX_MATCH = 2.5  # m; a farther prediction is not this player
# The top-view clicks are not at floor level: fitted on the dev matches (BD12-14, YOLO only) the annotated point
# sits ~0.32 m up, which puts every method's players ~0.15 m toward the net. GT_H=0.32 ray-casts the clicks at that
# height; the default 0 keeps the original floor-level GT (numbers comparable with earlier runs).
GT_H = float(os.environ.get("GT_H", "0"))
STABILIZE = os.environ.get("STABILIZE", "1") == "1"  # VIMO bodies through birdseye_v2.footlock.stabilize


def gt_players(m, r, top):
    out = {}
    for row in csv.DictReader(open(os.path.join(ROOT, "data", "datacsv", m, r, "1.csv"))):
        pts = []
        for n in (1, 2, 3, 4):
            x, y = row.get(f"player{n}x", ""), row.get(f"player{n}y", "")
            if x in ("", None) or y in ("", None) or float(x) < 0 or float(y) < 0:
                continue
            pts.append(top.pixel_to_ground(np.array([[float(x), float(y)]]), z=GT_H)[0][:2])
        if pts:
            out[int(row["frame"])] = np.array(pts)
    return out


def match_err(gt, pred):
    if not len(pred):
        return [None] * len(gt)
    C = np.linalg.norm(gt[:, None] - pred[None], axis=2)
    ri, ci = linear_sum_assignment(C)
    e = [None] * len(gt)
    for i, j in zip(ri, ci):
        if C[i, j] <= MAX_MATCH:
            e[i] = float(C[i, j])
    return e


def run(mr):
    m, r = mr
    try:
        cal = json.load(open(os.path.join(ROOT, "outputs", DRONE_CALIB, m, r + ".json")))
        top, back = Camera.from_dict(cal["top"]), Camera.from_dict(cal["back"])
        g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
        if not g.get("sync"):
            return None
        off = g["sync"]["offset"]
        poses = load_pose(os.path.join(ROOT, "outputs", "pose", m, r + ".npz"))
        gt = gt_players(m, r, top)
        frames = np.arange(len(poses))
        tracks = [smooth_track(back, t) for t in track_players(back, poses, frames)]
        res = {"raw": [], "tracked": [], "vimo": []}
        vp = os.path.join(ROOT, "outputs", DRONE_VIMO, m, r + ".npz")
        vj = np.load(vp)["joints"] if os.path.exists(vp) else None  # (P,T,24,3)
        if vj is not None and STABILIZE:
            vj = np.stack([stabilize(J) for J in vj])
        for ft, P in gt.items():
            fb = int(round(ft + off))
            if not (0 <= fb < len(poses)):
                continue
            raw = np.array([p["foot"] for p in players_3d(back, poses[fb])]).reshape(-1, 2)
            trk = np.array([t["foot"][fb] for t in tracks if np.isfinite(t["foot"][fb]).all()]).reshape(-1, 2)
            res["raw"] += match_err(P, raw)
            res["tracked"] += match_err(P, trk)
            if vj is not None and fb < vj.shape[1]:
                feet = np.nanmean(vj[:, fb][:, [7, 8], :2], axis=1)  # SMPL mid-ankle: the point YOLO's foot uses (toes added ~3 cm)
                res["vimo"] += match_err(P, feet[np.isfinite(feet).all(1)].reshape(-1, 2))
        return {"match": m, "rally": r, **res}
    except Exception as e:
        print("FAILED", m, r, repr(e), flush=True)
        return None


def summary(errs):
    e = np.array([x for x in errs if x is not None])
    n = len(errs)
    return {"n_gt": n, "matched": round(len(e) / max(n, 1), 3), "median_m": round(float(np.median(e)), 3),
            "mean_m": round(float(e.mean()), 3), "p90_m": round(float(np.percentile(e, 90)), 3),
            "pct_within_0.8m_of_all_gt": round(100 * float((e < 0.8).sum()) / max(n, 1), 1)}


if __name__ == "__main__":
    tag = sys.argv[1]
    jobs = []
    for a in sys.argv[2:]:
        if "/" in a:
            jobs.append(tuple(a.split("/")))
        else:
            jobs += [(a, os.path.basename(p)[:-5]) for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "gt", a, "*.json")))]
    with ProcessPoolExecutor(14) as ex:
        res = [x for x in ex.map(run, jobs) if x]
    keys = ("raw", "tracked") + (("vimo",) if any(x["vimo"] for x in res) else ())
    if "vimo" in keys:  # compare on the rallies that have VIMO only
        res = [x for x in res if x["vimo"]]
    out = {k: summary([e for x in res for e in x[k]]) for k in keys}
    out["rallies"] = len(res)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(ROOT, "outputs", "eval", f"players_{tag}.json"), "w"), indent=1)
