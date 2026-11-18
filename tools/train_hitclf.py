"""Train the learned back-view hit detector (birdseye_v2.hitclf) on dev annotations, pick the
probability threshold by cross-validated F1 on dev, report hit P/R/F1 on test once.
usage: python tools/train_hitclf.py   -> outputs/hitclf.joblib"""
import csv, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import Camera
from birdseye_v2.hitclf import FEATURES, candidate_features
from birdseye_v2.players import load_pose
from birdseye_v2.shuttle3d import clean_track, landing_frame

DEV = ["BD12", "BD13", "BD14"]
TEST = ["BD15", "BD16", "BD17", "BD18", "BD19", "BD20", "BD21", "BD22", "BD23"]


def rally_data(mr):
    m, r = mr
    try:
        g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
        if not g["sync"] or len(g["back_track"]["frames"]) < 10:
            return None
        off = g["sync"]["offset"]
        fr = np.array(g["back_track"]["frames"], float)
        px = np.array(g["back_track"]["px"], float).reshape(-1, 2)
        lf = landing_frame(fr, px)
        fr, px = clean_track(fr, px, [lf - 1] if lf is not None else [fr[-1]])
        cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "calib", m, r + ".json")))["back"])
        poses = load_pose(os.path.join(ROOT, "outputs", "pose", m, r + ".npz"))
        hits = [int(x["frame"]) + off for x in csv.DictReader(open(os.path.join(ROOT, "data", "datacsv", m, r, "1.csv")))
                if x["status"] not in ("0", "")]
        cands = candidate_features(cam, fr, px, poses)
        if lf is not None:
            cands = [c for c in cands if c["tau"] < lf - 2]
        hits = [h for h in hits if fr[0] - 2 <= h <= (lf if lf is not None else fr[-1])]
        return {"match": m, "rally": r, "hits": hits, "cands": cands}
    except Exception as e:
        print("FAILED", m, r, repr(e), flush=True)
        return None


def load(matches):
    jobs = [(m, os.path.basename(p)[:-5]) for m in matches for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "gt", m, "*.json")))]
    with ProcessPoolExecutor(16) as ex:
        return [d for d in ex.map(rally_data, jobs) if d]


def xy(data):
    X, y, grp = [], [], []
    for k, d in enumerate(data):
        for c in d["cands"]:
            X.append([c[f] for f in FEATURES])
            y.append(int(any(abs(c["tau"] - h) <= 3 for h in d["hits"])))
            grp.append(k)
    return np.array(X), np.array(y), np.array(grp)


def prf(data, probs_by_rally, thr, nms=6):
    tp = fp = fn = 0
    for d, p in zip(data, probs_by_rally):
        det = []
        for i in np.argsort(-p):
            if p[i] >= thr and all(abs(d["cands"][i]["tau"] - h) >= nms for h in det):
                det.append(d["cands"][i]["tau"])
        used = set()
        for h in d["hits"]:
            c = [(abs(x - h), x) for x in det if abs(x - h) <= 3 and x not in used]
            if c:
                tp += 1; used.add(min(c)[1])
            else:
                fn += 1
        fp += len(det) - len(used)
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


def probs(model, data):
    return [model.predict_proba(np.array([[c[f] for f in FEATURES] for c in d["cands"]]))[:, 1] if d["cands"] else np.zeros(0)
            for d in data]


if __name__ == "__main__":
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    dev, test = load(DEV), load(TEST)
    X, y, grp = xy(dev)
    print("dev candidates", len(y), "positives", int(y.sum()))
    # threshold by 5-fold CV over rallies on dev
    oof = [None] * len(dev)
    for tr, va in GroupKFold(5).split(X, y, grp):
        mdl = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15).fit(X[tr], y[tr])
        for k in set(grp[va]):
            d = dev[k]
            oof[k] = mdl.predict_proba(np.array([[c[f] for f in FEATURES] for c in d["cands"]]))[:, 1] if d["cands"] else np.zeros(0)
    best = max((prf(dev, oof, t)[2], t) for t in np.arange(0.15, 0.8, 0.05))
    thr = float(best[1])
    print("dev CV F1 %.3f at threshold %.2f" % best)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15).fit(X, y)
    P, R, F = prf(test, probs(model, test), thr)
    print(f"TEST hits: P={P:.3f} R={R:.3f} F1={F:.3f}")
    joblib.dump({"model": model, "threshold": thr, "features": FEATURES}, os.path.join(ROOT, "outputs", "hitclf.joblib"))
