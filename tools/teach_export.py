"""Confident teacher labels for the student (tools/train_rallynet.py TEACH=1 TEACH_DIR=<dir>).

The teacher (tools/shot_fit.py on all four cameras + tools/hit_decode.py hits) abstains where it is unsure, per label:
  3D shuttle  a frame is kept when its worst camera residual is small (res < TX_RES sigma) and its shot as a whole
              explains the cameras (median residual over observed frames < TX_SHOT sigma)
  hits        a hit is a positive when the video model is sure (p >= TX_HIT_HI), or fairly sure (p >= TX_HIT_LO) and
              the flights on both sides of it fit the cameras (the physics confirms a contact there); other decoded
              hits are masked out (+-TX_PAD frames: neither hit nor no-hit)
Writes outputs/cctv/<dir>/<uid>.json {S: [[frame, x, y, z]], hits: [frame], hit_mask: [[a, b]] (masked spans),
conf_frac, n_shots} -- the format train_rallynet reads, plus the masked spans.

usage: python tools/teach_export.py <out dir name> [uid ...]   (default: every rally in outputs/cctv/shot_fit_all)
       python tools/teach_export.py --eval <answer key json>   coverage + precision of the confident hits"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SF = os.path.join(ROOT, "outputs", "cctv", os.environ.get("TX_SF", "shot_fit_all"))
HITS = os.path.join(ROOT, "outputs", "cctv", "hits_all.json")
TX_RES = float(os.environ.get("TX_RES", "3.0"))
TX_SHOT = float(os.environ.get("TX_SHOT", "2.0"))
TX_HIT_HI = float(os.environ.get("TX_HIT_HI", "0.95"))
TX_HIT_LO = float(os.environ.get("TX_HIT_LO", "0.5"))
TX_PAD = int(os.environ.get("TX_PAD", "3"))


def shot_quality(o):
    res, shot = np.array(o["res"], float), np.array(o["shot"], int)
    q = {}
    for i in np.unique(shot):
        r = res[(shot == i) & (res > 0)]
        q[int(i)] = float(np.median(r)) if len(r) >= 3 else np.inf
    return q


def export(uid, H):
    o = json.load(open(os.path.join(SF, uid + ".json")))
    frames, P = np.array(o["frames"], int), np.array(o["P"], float)
    res, shot = np.array(o["res"], float), np.array(o["shot"], int)
    q = shot_quality(o)
    good = np.array([q[int(s)] < TX_SHOT for s in shot]) & (res < TX_RES) & np.isfinite(P).all(1)
    good &= np.nan_to_num(P[:, 2], nan=-1) > -0.05  # the shuttle never goes below the floor (the last flight ends at landing)
    S = [[int(f), *np.round(P[j], 3).tolist()] for j, f in enumerate(frames) if good[j]]
    prob = {int(h[0]): float(h[2]) for h in H.get(uid, [])}
    hits, mask = [], []
    for i, c in enumerate(o["contacts"]):
        f = int(c[0]); p = prob.get(f, 0.0)
        fits = q.get(i, np.inf) < TX_SHOT and (i == 0 or q.get(i - 1, np.inf) < TX_SHOT)
        if p >= TX_HIT_HI or (p >= TX_HIT_LO and fits):
            hits.append(f)
        else:
            mask.append([f - TX_PAD, f + TX_PAD])
    return {"S": S, "hits": hits, "hit_mask": mask, "conf_frac": round(float(good.mean()), 3) if len(good) else 0.0,
            "n_shots": len(o["contacts"])}


def evaluate(key_path):
    """confident hits vs the hand-labelled answer key: how many labelled hits the teacher keeps (coverage), how many of
    its confident hits are real (precision), and how many labelled hits fall inside a masked span (abstained)"""
    G = json.load(open(key_path)); H = json.load(open(HITS))
    n = kept = pos = good_pos = abst = 0; fr = []
    for u, g in G.items():
        if not os.path.exists(os.path.join(SF, u + ".json")):
            continue
        t = export(u, H); gf = [f for f, _ in g]; n += len(gf)
        kept += sum(any(abs(h - f) <= 2 for h in t["hits"]) for f in gf)
        abst += sum(any(a <= f <= b for a, b in t["hit_mask"]) for f in gf)
        pos += len(t["hits"]); good_pos += sum(any(abs(h - f) <= 2 for f in gf) for h in t["hits"])
        fr.append(t["conf_frac"])
    print(f"rallies {len(fr)}  labelled hits {n}: confident {kept} ({kept / max(n, 1):.0%}), abstained {abst}, "
          f"missed {n - kept - abst} | confident hits {pos}, precision {good_pos / max(pos, 1):.1%} | "
          f"3D frames kept {np.mean(fr):.0%}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--eval":
        evaluate(a[1]); sys.exit()
    out = os.path.join(ROOT, "outputs", "cctv", a[0]); os.makedirs(out, exist_ok=True)
    uids = a[1:] or sorted(f[:-5] for f in os.listdir(SF))
    H = json.load(open(HITS)); cf = []
    for u in uids:
        t = export(u, H); json.dump(t, open(os.path.join(out, u + ".json"), "w")); cf.append(t["conf_frac"])
    print(len(uids), "rallies, 3D frames kept", f"{np.mean(cf):.0%}")
