"""Export a few rallies for the 3D demo page: GT points, v1 and v2 trajectories (oracle hits), hits.
usage: python tools/export_demo.py out.json BD15/Rally-39 BD16/Rally-9 ..."""
import csv, json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import eval_shuttle as E
from birdseye_v2.court import Camera
from birdseye_v2.pipeline import reconstruct_rally
from birdseye_v2.players import load_pose

out = []
for mr in sys.argv[2:]:
    m, r = mr.split("/")
    g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
    off = g["sync"]["offset"]
    cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "calib", m, r + ".json")))["back"])
    poses = load_pose(os.path.join(ROOT, "outputs", "pose", m, r + ".npz"))
    hits = [int(x["frame"]) + off for x in csv.DictReader(open(os.path.join(ROOT, "data", "datacsv", m, r, "1.csv")))
            if x["status"] not in ("0", "")]
    rec = reconstruct_rally(cam, g["back_track"]["frames"], g["back_track"]["px"], poses, hits=hits)
    fr, px = rec["frames"], rec["px"]
    t = np.arange(rec["shots"][0][0], fr[-1] + 1e-9, 0.5)
    v2 = rec["fit"].position(t, rest_until=fr[-1])
    hands = [E.hands_for(cam, poses, fr, px, a) for a, _ in rec["shots"]]
    v1 = E.fit_v1(cam, fr, px, rec["shots"], hands)(t)
    gt_t = np.array([p["frame"] for p in g["points"]]) + off
    gt_X = np.array([p["X"] for p in g["points"]])
    ev = lambda P: np.linalg.norm(P - gt_X, axis=1)
    e2 = ev(rec["fit"].position(gt_t, rest_until=fr[-1]))
    e1 = ev(E.fit_v1(cam, fr, px, rec["shots"], hands)(gt_t))
    rnd = lambda a: [None if np.isnan(x).any() else [round(float(v), 3) for v in x] for x in a]
    out.append({"id": f"{m} {r}", "t": [round(float(x), 2) for x in t], "v2": rnd(v2), "v1": rnd(v1),
                "gt_t": [round(float(x), 2) for x in gt_t], "gt": rnd(gt_X), "hits": [round(float(h), 2) for h in hits],
                "err_v2_median": round(float(np.nanmedian(e2)), 3),
                "err_v1_median": round(float(np.nanmedian(e1)), 3) if np.isfinite(e1).any() else None,
                "v1_coverage": round(float(np.isfinite(e1).mean()), 3)})
    print(m, r, out[-1]["err_v1_median"], out[-1]["err_v2_median"])
json.dump(out, open(sys.argv[1], "w"))
