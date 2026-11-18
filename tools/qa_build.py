"""Build the rally QA benchmark (todo priority 3): question/answer pairs whose answers come from
DroneData ground truth (hit labels + two-view 3D shuttle GT), paired with a SCENE built only from
the back view by the v2 pipeline (detected hits, no labels). An answerer (rule-based or an LLM,
tools/qa_run.py) sees the scene + question, never the GT.

Output: outputs/qa/<tag>.jsonl, one rally per line:
  {match, rally, scene: {...}, questions: [{id, question, answer, kind, tol}]}
usage: python tools/qa_build.py --tag test BD15 BD16"""
import argparse, csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import Camera, NET_Y
from birdseye_v2.pipeline import rally_summary, reconstruct_rally
from birdseye_v2.players import load_pose

GT = os.path.join(ROOT, "outputs", "gt")


def gt_facts(g, hits):
    """Answers from ground truth. Hit frames are top-view clock; GT points too."""
    f = np.array([p["frame"] for p in g["points"]], float)
    X = np.array([p["X"] for p in g["points"]]).reshape(-1, 3)
    if len(f) < 10 or len(hits) < 1:
        return None

    def at(t):  # GT position nearest in time (within 3 frames)
        i = np.argmin(np.abs(f - t))
        return X[i] if abs(f[i] - t) <= 3 else None

    last = at(hits[-1])
    tail = X[f > hits[-1]]
    land = tail[-1] if len(tail) and tail[-1][2] < 0.5 else None
    seg_max = []
    for a, b in zip(hits, hits[1:] + [f[-1] + 1]):
        m = (f > a) & (f < b)
        seg_max.append(float(X[m, 2].max()) if m.any() else -1)
    return {
        "n_hits": len(hits),
        "last_hitter_side": None if last is None else ("near" if last[1] < NET_Y else "far"),
        "landing_half": None if land is None else ("near" if land[1] < NET_Y else "far"),
        "max_height_m": float(X[:, 2].max()),
        "highest_shot": int(np.argmax(seg_max)) + 1 if max(seg_max) > 0 else None,
    }


QUESTIONS = [
    ("n_hits", "How many times was the shuttle hit in this rally (count every stroke, including the serve)?",
     "int", 0),
    ("last_hitter_side", "Which side of the court played the final stroke of the rally: 'near' (the half closest "
     "to the camera, y < 6.7 m) or 'far'?", "choice", None),
    ("landing_half", "In which half of the court did the shuttle finally land: 'near' (y < 6.7 m) or 'far'?",
     "choice", None),
    ("max_height_m", "What was the maximum height of the shuttle above the floor during the rally, in metres?",
     "float", 0.5),
    ("highest_shot", "Which stroke number (1 = serve) sent the shuttle highest?", "int", 0),
]


def build(job):
    m, r = job
    try:
        g = json.load(open(os.path.join(GT, m, r + ".json")))
        if not g["sync"]:
            return None
        hits = [int(x["frame"]) for x in csv.DictReader(open(os.path.join(ROOT, "data", "datacsv", m, r, "1.csv")))
                if x["status"] not in ("0", "")]
        facts = gt_facts(g, hits)
        if facts is None:
            return None
        cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "calib", m, r + ".json")))["back"])
        poses = load_pose(os.path.join(ROOT, "outputs", "pose", m, r + ".npz"))
        hm = None
        if os.environ.get("QA_HITCLF") == "1":  # learned hit detector
            import joblib
            hm = joblib.load(os.path.join(ROOT, "outputs", "hitclf.joblib"))
        rec = reconstruct_rally(cam, g["back_track"]["frames"], g["back_track"]["px"], poses, hit_model=hm)  # detected hits
        fit = rec["fit"]
        tail = fit.position([rec["frames"][-1]], rest_until=rec["frames"][-1])[0]
        scene = {
            "court": {"frame": "metres; x across 0..6.1 (left->right seen from the back camera), y along 0..13.4 "
                               "(0 = baseline nearest the camera), z up; net at y=6.7, 1.55 m high"},
            "shots": rally_summary(rec),
            "final_shuttle_xyz_m": np.round(tail, 2).tolist(),
            "landing_detected": rec["landing"] is not None,
        }
        qs = [{"id": k, "question": q, "answer": facts[k], "kind": kind, "tol": tol}
              for k, q, kind, tol in QUESTIONS if facts[k] is not None]
        return {"match": m, "rally": r, "scene": scene, "questions": qs}
    except Exception as e:
        print("FAILED", m, r, repr(e), flush=True)
        return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("matches", nargs="+")
    a = ap.parse_args()
    jobs = [(m, os.path.basename(p)[:-5]) for m in a.matches for p in sorted(glob.glob(os.path.join(GT, m, "*.json")))
            if os.path.exists(os.path.join(ROOT, "outputs", "pose", m, os.path.basename(p)[:-5] + ".npz"))]
    os.makedirs(os.path.join(ROOT, "outputs", "qa"), exist_ok=True)
    n = 0
    with open(os.path.join(ROOT, "outputs", "qa", a.tag + ".jsonl"), "w") as fh, ProcessPoolExecutor(16) as ex:
        for res in ex.map(build, jobs):
            if res:
                fh.write(json.dumps(res) + "\n")
                n += 1
    print("rallies", n)
