"""Learned confidence for the dense teacher, from the self-hiding data of tools/teach_calib.py.

A gradient-boosted classifier predicts, per frame, whether the teacher's 3D is within OK_M of the truth, from the
teacher's own features (Laplace sds, distance to detections of each camera, jumps nearby, height, player distance,
speed). Cross-validated by rally; the label threshold is the one whose out-of-fold selected frames (hidden ones: what
the teacher filled in) have a 90th-percentile error <= BUDGET. Compared with thresholding the sd alone.
Output outputs/cctv/teach_conf.pkl {model, thr, feats}; prints the comparison and coverage on the full runs.
usage: python tools/teach_conf_fit.py   (env OK_M 0.15, BUDGET 0.15)"""
import glob, json, os, pickle, sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from teach_calib import FEATS, OUT  # noqa: E402

OK_M = float(os.environ.get("OK_M", "0.15"))
BUDGET = float(os.environ.get("BUDGET", "0.15"))
NF = len(FEATS)


def load():
    rows, full = [], {}
    for p in sorted(glob.glob(os.path.join(OUT, "*.npz"))):
        uid = os.path.basename(p)[:-4]
        z = np.load(p)
        for g, mode in enumerate(("side", "blocks")):
            a = z[mode]
            a = a[np.isfinite(a[:, NF])]
            rows.append(np.c_[a, np.full(len(a), g), np.full(len(a), hash(uid) % 100000)])
        full[uid] = z["full"]
    return np.concatenate(rows), full


def best_thr(score, err, budget):
    """Highest coverage threshold on score (higher = surer) with p90 error of the kept <= budget."""
    o = np.argsort(-score)
    e = err[o]
    p90 = np.array([np.percentile(e[:k], 90) if k >= 20 else 0 for k in range(1, len(e) + 1)])
    ok = np.nonzero(p90 <= budget)[0]
    k = ok.max() + 1 if len(ok) else 0
    return (score[o][k - 1] if k else np.inf), k / len(e)


def main():
    D, full = load()
    X, err, hid, mode, grp = D[:, :NF], D[:, NF], D[:, NF + 1] > 0, D[:, NF + 3], D[:, NF + 4]
    y = err < OK_M
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grp):
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                           monotonic_cst=[-1, -1, -1, -1, -1, 0, 0, 0, 0, 0])
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    for name, sel in (("hidden side-windows", hid & (mode == 0)), ("hidden blocks", hid & (mode == 1)), ("all hidden", hid)):
        t1, c1 = best_thr(-X[sel, 0], err[sel], BUDGET)
        t2, c2 = best_thr(oof[sel], err[sel], BUDGET)
        print(f"{name:22s} n={sel.sum():5d}  coverage at p90<={BUDGET}: sd alone {c1:.1%} (sd<{np.exp(-t1):.3f})   learned {c2:.1%} (p>{t2:.2f})")
    thr, cov = best_thr(oof[hid], err[hid], BUDGET)
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                       monotonic_cst=[-1, -1, -1, -1, -1, 0, 0, 0, 0, 0]).fit(X, y)
    pickle.dump({"model": m, "thr": float(thr), "feats": FEATS, "ok_m": OK_M, "budget": BUDGET},
                open(os.path.join(OUT, "..", os.path.basename(OUT) + "_conf.pkl"), "wb"))
    # coverage on the full runs (what becomes labels): in play = first to last teacher hit + 25 frames
    ca, cp = [], []
    for uid, a in full.items():
        p = m.predict_proba(a[:, :NF])[:, 1] >= thr
        fr = a[:, NF + 2]
        js = os.path.join(OUT, uid + "_full.json")
        hits = json.load(open(js))["hits"] if os.path.exists(js) else []
        ca.append(p.mean())
        if hits:
            k = (fr >= hits[0]) & (fr <= hits[-1] + 25)
            if k.any():
                cp.append(p[k].mean())
    print(f"full runs: {len(ca)} rallies, labelled {np.mean(ca):.1%} of frames, {np.mean(cp):.1%} of in-play frames (thr p>{thr:.2f})")


if __name__ == "__main__":
    main()
