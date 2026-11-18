"""Rally QA dataset v2 (for training a QA model, tools/qa2_train.py).

Per rally: a text SCENE built only from the back view by the v2 pipeline (detected hits, learned hit
detector) -- per-shot summary, the 3D shuttle fit sampled at 10 Hz, players' ground positions at 5 Hz --
and questions whose answers come from ground truth (hit labels + two-view 3D shuttle).

Question set: rally-level (stroke count, last hitter side, landing half, max height, highest stroke,
duration, strokes by the near side) and per-stroke (hitter side, apex, initial speed, hit downward?).

Output: outputs/qa2/<match>.jsonl, one rally per line:
  {match, rally, scene_text, shots, final_xyz, questions: [{id, q, answer, kind, tol}]}
usage: python tools/qa2_build.py BD12 BD13 ..."""
import csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import NET_Y, Camera
from birdseye_v2.pipeline import rally_summary, reconstruct_rally
from birdseye_v2.players import load_pose, players_3d

GT = os.path.join(ROOT, "outputs", "gt")
OUT = os.path.join(ROOT, "outputs", "qa2")
FPS = 30.0
MAX_STROKES = 12  # per-stroke questions for strokes 1..12


def side(y):
    return "near" if y < NET_Y else "far"


def gt_facts(f, X, hits):
    """f: GT frames (top clock), X: (N,3) GT points, hits: labeled hit frames (top clock)."""
    def at(t, w=3):
        i = np.argmin(np.abs(f - t))
        return X[i] if abs(f[i] - t) <= w else None

    facts, strokes = {}, []
    bounds = hits + [f[-1] + 1]
    for k, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        m = (f > a) & (f < b)
        s = {"side": None, "apex": None, "speed": None, "down": None}
        p = at(a)
        if p is None and m.any():
            p = X[np.where(m)[0][0]] if f[np.where(m)[0][0]] - a <= 6 else None
        if p is not None:
            s["side"] = side(p[1])
        if m.sum() >= 0.5 * max(b - a - 1, 1) and m.sum() >= 4:
            s["apex"] = float(X[m, 2].max())
        e = (f > a) & (f <= a + 5)
        if e.sum() >= 3:
            A = np.c_[f[e] - a, np.ones(e.sum())]
            v = np.linalg.lstsq(A, X[e], rcond=None)[0][0] * FPS  # m/s
            s["speed"] = float(np.linalg.norm(v) * 3.6)
            s["down"] = "yes" if v[2] < -0.5 else "no"
        strokes.append(s)

    # sides strictly alternate (every stroke crosses the net): pick the serve side that agrees best
    # with the per-stroke GT measurements, which are noisy right at the hit (player occludes it)
    meas = [st["side"] for st in strokes]
    alt = lambda s0: [s0 if i % 2 == 0 else ("far" if s0 == "near" else "near") for i in range(len(strokes))]
    score = {s0: sum(a == b for a, b in zip(alt(s0), meas) if a and b) for s0 in ("near", "far")}
    n_meas = sum(m is not None for m in meas)
    s0 = max(score, key=score.get)
    ok = n_meas >= 2 and score[s0] >= 0.75 * n_meas
    for st, sd in zip(strokes, alt(s0)):
        st["side"] = sd if ok else None

    tail = X[f > hits[-1]]
    land = tail[-1] if len(tail) and tail[-1][2] < 0.5 else None
    seg_max = [st["apex"] if st["apex"] is not None else -1 for st in strokes]
    facts["n_hits"] = len(hits)
    facts["last_hitter_side"] = strokes[-1]["side"]
    facts["landing_half"] = None if land is None else side(land[1])
    facts["max_height_m"] = float(X[:, 2].max())
    facts["highest_shot"] = int(np.argmax(seg_max)) + 1 if max(seg_max) > 0 and all(a is not None for a in seg_max) else None
    facts["duration_s"] = float((f[-1] - hits[0]) / FPS) if land is not None else None
    sides = [st["side"] for st in strokes]
    facts["near_strokes"] = sum(sd == "near" for sd in sides) if all(sides) else None
    return facts, strokes


QUESTIONS = [  # id, text, kind, tol
    ("n_hits", "How many times was the shuttle hit in this rally (count every stroke, including the serve)?", "int", 0),
    ("last_hitter_side", "Which side played the final stroke of the rally: near or far?", "choice", None),
    ("landing_half", "In which half of the court did the shuttle finally land: near or far?", "choice", None),
    ("max_height_m", "What was the maximum height of the shuttle during the rally, in metres?", "float", 0.4),
    ("highest_shot", "Which stroke number (1 = serve) sent the shuttle highest?", "int", 0),
    ("duration_s", "How long was the rally from the serve until the shuttle landed, in seconds?", "float", 0.5),
    ("near_strokes", "How many strokes were played by the near side?", "int", 0),
]
STROKE_Q = [
    ("side", "Which side hit stroke {k}: near or far?", "choice", None),
    ("apex", "How high did the shuttle go during stroke {k}, in metres?", "float", 0.4),
    ("speed", "What was the shuttle's speed just after stroke {k}, in km/h?", "rel", 0.25),
    ("down", "Was stroke {k} hit downward (yes or no)?", "choice", None),
]


def scene_text(rec, cam, poses, t0):
    """Compact text rendering of the back-view reconstruction. Times in seconds from clip start."""
    fit = rec["fit"]
    lines = ["Court: x 0..6.1 m across, y 0..13.4 m along (y=0 is the near baseline, closest to the camera), "
             "z up. Net at y=6.7, 1.55 m high. near side = y<6.7."]
    lines.append("Detected shots (automatic, may miss or split strokes):")
    for s in rally_summary(rec):
        lines.append(f"  shot {s['shot']}: t={s['t_start_s'] - t0:.2f}-{s['t_end_s'] - t0:.2f}s hitter={s['hitter_side']}"
                     f"{'' if s['anchored'] else '(no player at hit)'} hit=({s['hit_xyz_m'][0]:.1f},{s['hit_xyz_m'][1]:.1f},"
                     f"{s['hit_xyz_m'][2]:.1f}) end=({s['end_xyz_m'][0]:.1f},{s['end_xyz_m'][1]:.1f},{s['end_xyz_m'][2]:.1f}) "
                     f"apex={s['apex_m']:.1f} speed={s['speed_kmh']:.0f}km/h")
    a, b = fit.shots[0][0], rec["frames"][-1]
    ts = np.arange(a, b + 1e-9, 3.0)
    P = fit.position(ts, rest_until=b)
    obs = rec["frames"]
    lines.append("Shuttle 3D track, 10 Hz (t: x y z; * = no 2D detection near this time):")
    lines.append("  " + " | ".join(f"{t / FPS - t0:.1f}: {p[0]:.1f} {p[1]:.1f} {p[2]:.1f}"
                                   + ("" if np.min(np.abs(obs - t)) <= 2 else "*")
                                   for t, p in zip(ts, P) if np.isfinite(p).all()))
    lines.append("Players on court, 5 Hz (t: near players x,y ; far players x,y):")
    pl = []
    for t in np.arange(a, b + 1e-9, 6.0):
        f = int(round(t))
        if not (0 <= f < len(poses)):
            continue
        ppl = players_3d(cam, poses[f])
        n = " ".join(f"{p['foot'][0]:.1f},{p['foot'][1]:.1f}" for p in ppl if p["foot"][1] < NET_Y)
        fa = " ".join(f"{p['foot'][0]:.1f},{p['foot'][1]:.1f}" for p in ppl if p["foot"][1] >= NET_Y)
        pl.append(f"{t / FPS - t0:.1f}: {n} ; {fa}")
    lines.append("  " + " | ".join(pl))
    return "\n".join(lines)


def build(job):
    m, r, hm_path = job
    try:
        g = json.load(open(os.path.join(GT, m, r + ".json")))
        if not g["sync"] or len(g["points"]) < 10:
            return None
        hits = [int(x["frame"]) for x in csv.DictReader(open(os.path.join(ROOT, "data", "datacsv", m, r, "1.csv")))
                if x["status"] not in ("0", "")]
        if not hits:
            return None
        f = np.array([p["frame"] for p in g["points"]], float)
        X = np.array([p["X"] for p in g["points"]]).reshape(-1, 3)
        facts, strokes = gt_facts(f, X, hits)
        cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "calib", m, r + ".json")))["back"])
        poses = load_pose(os.path.join(ROOT, "outputs", "pose", m, r + ".npz"))
        import joblib
        hm = joblib.load(hm_path)
        rec = reconstruct_rally(cam, g["back_track"]["frames"], g["back_track"]["px"], poses, hit_model=hm)
        t0 = rec["frames"][0] / FPS
        qs = [{"id": k, "q": q, "answer": facts[k], "kind": kind, "tol": tol}
              for k, q, kind, tol in QUESTIONS if facts[k] is not None]
        for k, st in enumerate(strokes[:MAX_STROKES]):
            for key, q, kind, tol in STROKE_Q:
                if st[key] is not None:
                    qs.append({"id": f"stroke_{key}", "k": k + 1, "q": q.format(k=k + 1),
                               "answer": round(st[key], 2) if isinstance(st[key], float) else st[key], "kind": kind, "tol": tol})
        tail = rec["fit"].position([rec["frames"][-1]], rest_until=rec["frames"][-1])[0]
        return {"match": m, "rally": r, "scene_text": scene_text(rec, cam, poses, t0),
                "shots": rally_summary(rec), "final_xyz": np.round(tail, 2).tolist(), "questions": qs}
    except Exception as e:
        print("FAILED", m, r, repr(e), flush=True)
        return None


if __name__ == "__main__":
    hm_path = os.path.join(ROOT, "outputs", "hitclf.joblib")
    os.makedirs(OUT, exist_ok=True)
    for m in sys.argv[1:]:
        jobs = [(m, os.path.basename(p)[:-5], hm_path) for p in sorted(glob.glob(os.path.join(GT, m, "*.json")))
                if os.path.exists(os.path.join(ROOT, "outputs", "pose", m, os.path.basename(p)[:-5] + ".npz"))]
        with ProcessPoolExecutor(16) as ex:
            res = [x for x in ex.map(build, jobs) if x]
        with open(os.path.join(OUT, m + ".jsonl"), "w") as fh:
            for x in res:
                fh.write(json.dumps(x) + "\n")
        print(m, "rallies", len(res), "questions", sum(len(x["questions"]) for x in res), flush=True)
