"""Hit detection on FineBadminton-20K rally clips (broadcast singles, frame-labelled hits; never used by
FBBench). Compares:
  v1      Birdseye v1's skeleton transformer (birdseye_v2.v1action) on VIMO bodies (tools/fb_vimo.py)
  clf     v2's learned shuttle-track detector (outputs/hitclf.joblib, trained on DroneData)
  rule    v2's rule-based shuttle-track detector
Matching: one-to-one, |frame error| <= tol (25 fps frames). tol=3 is our DroneData protocol;
tol=8 (~10 frames at 30 fps) is v1's own. Also reports hitter-side accuracy and, for v1, stroke type.
usage: python tools/hit_eval_fb.py uid ...   (or a file of uids with @file)"""
import json, os, sys

import joblib
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2 import v1action
from birdseye_v2.broadcast import FPS, native_frames, native_index, reconstruct_clip
from birdseye_v2.fbfeat import STROKE_NAMES

CLIPS = os.path.join(ROOT, "data", "fb20k", "clips")
OUT = os.path.join(ROOT, "outputs", "fb20k")


def match(pred, gt, tol):
    """pred, gt: lists of (frame, side, type|None). Greedy one-to-one by smallest error."""
    pairs = sorted((abs(p[0] - g[0]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p[0] - g[0]) <= tol)
    up, ug, m = set(), set(), []
    for _, i, j in pairs:
        if i not in up and j not in ug:
            up.add(i); ug.add(j); m.append((pred[i], gt[j]))
    return m


def main(uids):
    model = v1action.load()
    hm = joblib.load(os.path.join(ROOT, "outputs", "hitclf.joblib"))
    res = {k: {t: {"tp": 0, "fp": 0, "fn": 0, "side": 0, "type": 0, "type_n": 0} for t in (3, 8)} for k in ("v1", "clf", "rule")}
    n = 0
    for uid in uids:
        vp = os.path.join(OUT, os.environ.get("VIMO_DIR", "vimo_bc"), uid + ".npz")
        if not os.path.exists(vp):
            continue
        lab = json.load(open(os.path.join(CLIPS, uid + ".json"), encoding="utf8"))
        keep = native_frames(os.path.join(CLIPS, uid + ".mp4"), os.path.join(OUT, "native", uid + ".json"))
        gt = [(native_index(keep, h["frame"]), h["hitter"], STROKE_NAMES.get(h["type"], h["type"])) for h in lab["hits"]]
        v = np.load(vp)
        out = v1action.predict(model, v["top"], v["bottom"], src_fps=FPS)
        preds = {"v1": sorted([(p[0], s, p[2]) for s in ("top", "bottom") for p in out[s]["peaks"]])}
        video = os.path.join(CLIPS, uid + ".mp4")
        for k, hmod in (() if os.environ.get("V1_ONLY") else (("clf", hm), ("rule", None))):
            rec = reconstruct_clip(ROOT, uid, video, hit_model=hmod, out="fb20k")
            preds[k] = [] if rec is None else [(s["t_start_s"] * FPS, "bottom" if s["hitter_side"] == "near" else "top", None)
                                               for s in rec["summary"]]
        for k, pr in preds.items():
            for tol in (3, 8):
                mm = match(pr, gt, tol)
                r = res[k][tol]
                r["tp"] += len(mm); r["fp"] += len(pr) - len(mm); r["fn"] += len(gt) - len(mm)
                r["side"] += sum(p[1] == g[1] for p, g in mm)
                for p, g in mm:
                    if p[2] is not None and g[2] in v1action.TO_FB.values():
                        r["type_n"] += 1; r["type"] += p[2] == g[2]
        n += 1
    print("rallies", n)
    for k, d in res.items():
        for tol, r in d.items():
            P = r["tp"] / max(r["tp"] + r["fp"], 1); R = r["tp"] / max(r["tp"] + r["fn"], 1)
            F = 2 * P * R / max(P + R, 1e-9)
            line = f"{k:5s} tol={tol}: P={P:.3f} R={R:.3f} F1={F:.3f} side={r['side'] / max(r['tp'], 1):.3f}"
            if r["type_n"]:
                line += f" type={r['type'] / r['type_n']:.3f} (n={r['type_n']})"
            print(line, flush=True)
    json.dump(res, open(os.path.join(OUT, "hit_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0].startswith("@"):
        a = [l.strip() for l in open(a[0][1:]) if l.strip()]
    main(a)
