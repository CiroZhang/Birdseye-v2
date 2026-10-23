"""RallyNet's bodies for BFMD in Birdseye v1's layout (as tools/bfmd_export.py does for VIMO), so v1's builder and
hit + stroke trainer (tools/bfmd_stroke.py) measure how much the representation carries: same classifier, same
split, only the bodies change. RallyNet never trained on BFMD's test matches (tools/build_hit_labels.py).

COCO-17 joints are written into the SMPL-24 slots v1 expects: pelvis / spine / neck / collars interpolated from
hips and shoulders, feet = ankles, hands = wrists pushed 8 cm along the forearm. top = the far player's slot
(y > NET_Y) with most frames, bottom = the near one.

out  data/bfmd/<OUT_SD>/<match>/joints_selfdetected.json
usage: python tools/rallynet_bfmd_export.py   (env TAG=v7, OUT_SD=rn_sd)"""
import glob, json, os, sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from birdseye_v2 import rallynet as RN  # noqa: E402
from birdseye_v2.court import NET_Y  # noqa: E402
import train_rallynet as TR  # noqa: E402

DATA = os.path.join(ROOT, "data", "bfmd")
OUT_SD = os.path.join(DATA, os.environ.get("OUT_SD", "rn_sd"))
TAG = os.environ.get("TAG", "v7")


def smpl24(K):
    """(T,17,3) COCO -> (T,24,3) in SMPL joint order."""
    lh, rh, ls, rs = K[:, 11], K[:, 12], K[:, 5], K[:, 6]
    pel, neck = 0.5 * (lh + rh), 0.5 * (ls + rs)
    sp = lambda a: pel + a * (neck - pel)
    hand = lambda w, e: w + 0.08 * (w - e) / (np.linalg.norm(w - e, axis=-1, keepdims=True) + 1e-9)
    head = np.where(np.isfinite(K[:, :1, 0]), K[:, 0], neck + np.array([0, 0, 0.2]))
    return np.stack([pel, lh, rh, sp(0.3), K[:, 13], K[:, 14], sp(0.6), K[:, 15], K[:, 16], sp(0.85), K[:, 15], K[:, 16],
                     neck, 0.5 * (neck + ls), 0.5 * (neck + rs), head, ls, rs, K[:, 7], K[:, 8], K[:, 9], K[:, 10],
                     hand(K[:, 9], K[:, 7]), hand(K[:, 10], K[:, 8])], 1)


def main():
    model = TR.load_model(TAG)
    head = (lambda K: K) if TR.ARCH == "rallyphys" else RN.fix_head  # RallyPhys learns its face joints
    matches = open(os.path.join(DATA, "meta", "all_matches.txt")).read().split()
    for m in matches:
        rallies = json.load(open(os.path.join(DATA, "meta", m + ".rallies.json")))
        out = {}
        for r in rallies:
            tag = f"g{r['game']}_r{r['rally']:03d}"
            p = os.path.join(TR.OBS, "bfmd", f"{m}__{tag}.npz")
            if not os.path.exists(p):
                continue
            f = TR.load_feats(p)
            res = TR.infer(model, f)
            n = r["end"] - r["start"] + 1
            pres = f["present"] > 0
            joints = {}
            for side in ("top", "bottom"):
                best = None
                for q in range(RN.P):
                    if pres[:, q].sum() < 8:
                        continue
                    far = np.median(f["root"][pres[:, q], q, 1]) > NET_Y
                    if far == (side == "top") and (best is None or pres[:, q].sum() > pres[:, best].sum()):
                        best = q
                if best is None:
                    continue
                K = head(res["joints"][:, best])
                K[~pres[:, best]] = np.nan
                # the tracker can split one player into several slots: fill the gaps from the other slots on this side
                for q in sorted(range(RN.P), key=lambda q: -pres[:, q].sum()):
                    if q == best or pres[:, q].sum() < 8 or (np.median(f["root"][pres[:, q], q, 1]) > NET_Y) != (side == "top"):
                        continue
                    gap = ~np.isfinite(K[:, 0, 0]) & pres[:, q]
                    K[gap] = head(res["joints"][gap, q])
                J = np.full((n, 24, 3), np.nan)
                k = min(n, len(K))
                J[:k] = smpl24(K[:k])
                joints[side] = np.round(J, 4).tolist()
            out[tag] = {"start": r["start"], "end": r["end"], "joints": joints}
        os.makedirs(os.path.join(OUT_SD, m), exist_ok=True)
        json.dump({"rallies": out}, open(os.path.join(OUT_SD, m, "joints_selfdetected.json"), "w"))
        print(f"{m}: {len(out)}/{len(rallies)} rallies", flush=True)


if __name__ == "__main__":
    main()
