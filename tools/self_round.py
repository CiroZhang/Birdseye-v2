"""One round of the self-learning loop: the student labels frames the teacher could not, where it can be trusted.

1. Run the TV-only student (rallyphys STUDENT=1, current env + TAG) on CCTV rallies; per frame, its own diagnostics:
   physics residuals (velocity change not explained by gravity + the learned drag, position vs velocity), hit gate,
   TV visibility, distance to the fixed side camera's shuttle ray where the side view sees it, height, speed,
   distance to the nearest predicted hit.
2. Calibration (rallies NOT used to train the student, --calib): a gradient-boosted classifier predicts from these
   whether the student is within OK_M of the teacher; its threshold is the lowest whose selected frames keep a
   90th-percentile error <= BUDGET (cross-validated by rally) -- the same rule as the teacher's own confidence.
3. Labelling (--uids, training rallies): frames the teacher left unlabelled and the classifier accepts are added.
   Output OUT_DIR/<uid>.json = the teacher's file (TEACH_DIR) with S extended by the accepted student frames and
   "S_student": the added frame indices. Train the next round with TEACH_DIR=<OUT_DIR basename>.

usage: env <student env> TAG=<tag> python tools/self_round.py OUT_DIR --calib calib_uids.txt --uids train_uids.txt
       [--cache feats.pkl]   (env OK_M 0.15, BUDGET 0.15)
"""
import argparse, json, os, pickle, sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_rallynet as TR  # noqa: E402
from birdseye_v2.rallynet import G  # noqa: E402

OK_M = float(os.environ.get("OK_M", "0.15"))
BUDGET = float(os.environ.get("BUDGET", "0.15"))
TEACH_DIR = os.environ.get("TEACH_DIR", "teach_v3")
FEATS = ["dv", "dx", "gate", "tv_vis", "side_vis", "side_d", "z", "speed", "hit_dist"]


def rally_feats(M, uid):
    f = TR.load_feats(os.path.join(TR.OBS, "cctv", uid + ".npz"))
    out = TR.infer(M, f)
    S, V = out["shuttle"], out["svel"]
    T, dt = len(S), float(f["dt"])
    with torch.no_grad():
        a = M.aero(torch.tensor(V[:-1], dtype=torch.float32, device=next(M.parameters()).device)).cpu().numpy()
    g = np.array([0, 0, -G])
    dv = np.linalg.norm(V[1:] - V[:-1] - (g + a) * dt, axis=1)
    dx = np.linalg.norm(S[1:] - S[:-1] - 0.5 * (V[1:] + V[:-1]) * dt, axis=1)
    pad = lambda r: np.maximum(np.r_[r, r[-1]], np.r_[r[0], r])  # per frame: worst of the two adjacent steps
    gate = out["gate"] if "gate" in out else 1 - np.prod(1 - 1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    hp = 1 - np.prod(1 - 1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    hits = np.array(TR.hits_from(hp, nms=8), int)
    hd = np.min(np.abs(np.arange(T)[:, None] - hits[None]), 1) if len(hits) else np.full(T, 99)
    C, D, vis = TR.side_rays(uid, T)
    w = S - C
    side_d = np.where(vis > 0, np.linalg.norm(w - (w * D).sum(1, keepdims=True) * D, axis=1), 0.0)
    X = np.c_[pad(dv), pad(dx), gate, f["sh_vis"], vis, side_d, S[:, 2], np.linalg.norm(V, axis=1), np.minimum(hd, 99)]
    ok = (f["cam_ok"] > 0) & np.isfinite(X).all(1) & np.isfinite(S).all(1)
    return X.astype(np.float32), S, ok


def teacher(uid):
    p = os.path.join(ROOT, "outputs", "cctv", TEACH_DIR, uid + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--calib", required=True); ap.add_argument("--uids", required=True)
    ap.add_argument("--cache")
    a = ap.parse_args()
    cal = open(a.calib).read().split(); tr = [u for u in open(a.uids).read().split() if u not in set(cal)]
    data = pickle.load(open(a.cache, "rb")) if a.cache and os.path.exists(a.cache) else {}
    todo = [u for u in cal + tr if u not in data]
    if todo:
        M = TR.load_model()
        for n, uid in enumerate(todo):
            try:
                data[uid] = rally_feats(M, uid)
            except Exception as e:  # missing observations etc.
                print(uid, "skip", e, flush=True); data[uid] = None
            if n % 50 == 0:
                print(n, len(todo), uid, flush=True)
                if a.cache:
                    pickle.dump(data, open(a.cache, "wb"))
        if a.cache:
            pickle.dump(data, open(a.cache, "wb"))
    # ---- calibration on held-out rallies: is the student within OK_M of the teacher?
    Xc, ec, gc = [], [], []
    for gi, uid in enumerate(cal):
        tj = teacher(uid)
        if data.get(uid) is None or tj is None:
            continue
        X, S, ok = data[uid]
        for k, x, y, z in tj["S"]:
            if 0 <= k < len(S) and ok[k]:
                Xc.append(X[k]); ec.append(np.linalg.norm(S[k] - (x, y, z))); gc.append(gi)
    Xc, ec, gc = np.array(Xc), np.array(ec), np.array(gc)
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    pr = np.zeros(len(ec))
    for tri, tei in GroupKFold(n_splits=min(5, len(set(gc)))).split(Xc, ec < OK_M, gc):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15).fit(Xc[tri], ec[tri] < OK_M)
        pr[tei] = clf.predict_proba(Xc[tei])[:, 1]
    thr = None
    for t in np.linspace(0.05, 0.95, 91):
        sel = pr >= t
        if sel.sum() >= 50 and np.percentile(ec[sel], 90) <= BUDGET:
            thr = float(t); break
    print("calibration: %d frames from %d rallies, student median err %.3f m, p90 %.3f" % (len(ec), len(set(gc)), np.median(ec), np.percentile(ec, 90)))
    if thr is None:
        print("no threshold keeps p90 <= %.2f m: nothing accepted" % BUDGET); return
    sel = pr >= thr
    print("threshold %.2f: keeps %.1f%% of held-out frames, their median %.3f p90 %.3f" % (thr, 100 * sel.mean(), np.median(ec[sel]), np.percentile(ec[sel], 90)))
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15).fit(Xc, ec < OK_M)
    # ---- label the training rallies' unlabelled frames
    os.makedirs(a.out, exist_ok=True)
    n_t = n_add = 0
    for uid in tr:
        tj = teacher(uid)
        if tj is None or data.get(uid) is None:
            continue
        X, S, ok = data[uid]
        have = {int(r[0]) for r in tj["S"]}
        cand = np.array([k for k in range(len(S)) if ok[k] and k not in have], int)
        add = cand[clf.predict_proba(X[cand])[:, 1] >= thr] if len(cand) else cand
        tj = dict(tj)
        tj["S"] = sorted(tj["S"] + [[int(k), *np.round(S[k], 3).tolist()] for k in add], key=lambda r: r[0])
        tj["S_student"] = add.tolist()
        json.dump(tj, open(os.path.join(a.out, uid + ".json"), "w"))
        n_t += len(have); n_add += len(add)
    print("labels: teacher %d frames, student adds %d (+%.1f%%)" % (n_t, n_add, 100 * n_add / max(n_t, 1)))
    json.dump({"thr": thr, "OK_M": OK_M, "BUDGET": BUDGET, "feats": FEATS, "teacher": n_t, "added": n_add},
              open(os.path.join(a.out, "_round.json"), "w"))


if __name__ == "__main__":
    main()
