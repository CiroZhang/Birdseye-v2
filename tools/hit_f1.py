"""Hit P/R/F1 of an eval_shuttle run (outputs/eval/<tag>.json; runs since the joint mode store their hits):
predicted hits after the track start vs annotated hits inside the track, one-to-one within TOL frames.
usage: python tools/hit_f1.py tag [tag ...]   (env TOL=3)"""
import json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOL = float(os.environ.get("TOL", "3"))


def prf(rallies, tol=TOL):
    tp = fp = fn = 0
    for r in rallies:
        t0, t1 = r["track"]
        pred = [h for h in r["hits_pred"] if h > t0 + 0.5]
        gt = [h for h in r["hits_gt"] if t0 + 2 <= h <= t1]
        used = set()
        for h in gt:
            c = [(abs(p - h), k) for k, p in enumerate(pred) if abs(p - h) <= tol and k not in used]
            if c:
                tp += 1; used.add(min(c)[1])
            else:
                fn += 1
        fp += len(pred) - len(used)
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9)


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        d = json.load(open(os.path.join(ROOT, "outputs", "eval", tag + ".json")))
        rs = [r for r in d["rallies"] if "hits_pred" in r]
        line = " ".join(f"{n}={v['median']:.3f}/{v['mean']:.3f}" for n, v in d["table"].items())
        print(f"{tag:28s} n={len(rs)} " + " ".join("F1@%d %.3f (P %.2f R %.2f)" % (t, *prf(rs, t)[::-1][:1], *prf(rs, t)[:2]) for t in (3, 5)) + "  3D " + line)
