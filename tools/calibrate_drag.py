"""Shuttle drag from REAL 3D data: fit gravity+quadratic-drag flights (free terminal velocity) to
the two-view GT points of every well-observed shot (between consecutive annotated hits).
Birdseye v1 could only calibrate drag through 2D reprojection on broadcast video.
usage: python tools/calibrate_drag.py [BD12 ...]   -> outputs/drag_calibration.json"""
import csv, glob, json, os, sys

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.shuttle3d import simulate

GT = os.path.join(ROOT, "outputs", "gt")
CSV_DIR = os.path.join(ROOT, "data", "datacsv")


def shots_from_gt(m, r):
    g = json.load(open(os.path.join(GT, m, r + ".json")))
    hits = [int(x["frame"]) for x in csv.DictReader(open(os.path.join(CSV_DIR, m, r, "1.csv")))
            if x["status"] not in ("0", "")]
    f = np.array([p["frame"] for p in g["points"]], float)
    X = np.array([p["X"] for p in g["points"]]).reshape(-1, 3)
    for a, b in zip(hits[:-1], hits[1:]):
        msk = (f > a + 1) & (f < b - 1)  # stay clear of the racket contact
        if msk.sum() >= 10 and f[msk].ptp() >= 12:
            yield m, r, f[msk], X[msk]


def fit(f, X, vterm=None):
    t = (f - f[0]) / 30.0
    v0 = (X[min(3, len(X) - 1)] - X[0]) / max(t[min(3, len(t) - 1)], 1e-3)
    if vterm is None:
        res = least_squares(lambda p: (simulate(p[:3], p[3:6], t, k=9.81 / p[6] ** 2) - X).ravel(),
                            np.r_[X[0], v0, 6.5], bounds=([-np.inf] * 6 + [2.0], [np.inf] * 6 + [30.0]))
    else:
        res = least_squares(lambda p: (simulate(p[:3], p[3:6], t, k=9.81 / vterm ** 2) - X).ravel(), np.r_[X[0], v0])
    return res.x, np.sqrt(np.mean(res.fun ** 2))


if __name__ == "__main__":
    matches = sys.argv[1:] or sorted(os.listdir(GT))
    rows = []
    for m in matches:
        for gj in sorted(glob.glob(os.path.join(GT, m, "*.json"))):
            r = os.path.basename(gj)[:-5]
            try:
                for _, _, f, X in shots_from_gt(m, r):
                    p, rms = fit(f, X)
                    _, rms65 = fit(f, X, 6.5)
                    rows.append({"match": m, "rally": r, "n": int(len(f)), "dur_s": float(f.ptp() / 30),
                                 "vterm": float(p[6]), "speed0": float(np.linalg.norm(p[3:6])),
                                 "rms": float(rms), "rms_v65": float(rms65)})
            except Exception as e:
                print("skip", m, r, repr(e))
    v = np.array([x["vterm"] for x in rows]); rms = np.array([x["rms"] for x in rows])
    good = rms < 0.08  # shots the drag model explains well
    print(f"shots {len(rows)}  good-fit {good.sum()}")
    print(f"v_terminal median {np.median(v[good]):.2f} m/s  IQR {np.percentile(v[good], 25):.2f}-{np.percentile(v[good], 75):.2f}")
    print(f"rms free-vterm median {np.median(rms):.3f} m   fixed 6.5 median {np.median([x['rms_v65'] for x in rows]):.3f} m")
    h, e = np.histogram(v[good], bins=np.arange(2, 16.1, 0.5))
    for c, a in zip(h, e[:-1]):
        print(f"{a:5.1f} {'#' * c}")
    json.dump({"rows": rows, "vterm_median": float(np.median(v[good]))},
              open(os.path.join(ROOT, "outputs", "drag_calibration.json"), "w"), indent=1)
