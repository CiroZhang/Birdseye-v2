"""The demo viewer's data.json (rendered by docs/index.html) from RallyNet
alone: skeletons, rackets, shuttle and hits all come out of the one model; the four detectors are its only input.
Ground truth (two-view shuttle, top-view players, labelled hits) is drawn for comparison only.

usage: python tools/export_rallynet_viewer.py BD15/Rally-39 ...   (env TAG=v0, VIEWER_OUT=viewer_rallynet)"""
import json, os, sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from birdseye_v2 import rallynet as RN  # noqa: E402
from birdseye_v2.court import NET_Y, Camera  # noqa: E402
from birdseye_v2.players import load_rackets  # noqa: E402
from birdseye_v2.tracks import RACKET_HEAD, RACKET_LEN  # noqa: E402
import train_rallynet as TR  # noqa: E402
from viewer_util import face_normals, norm_px, r3, write_clip  # noqa: E402

OUT = os.path.join(ROOT, "outputs", os.environ.get("VIEWER_OUT", "viewer_rallynet"))
TAG = os.environ.get("TAG", "v0")
PRE, POST = 15, 30
GRIP = 0.08


def export(mr, model):
    m, r = mr.split("/")
    p = os.path.join(TR.OBS, "drone", f"{m}__{r}.npz")
    gt = json.load(open(p[:-4] + "_gt.json"))
    cam = Camera.from_dict(gt["cam"])
    f = TR.load_feats(p)
    out = TR.infer(model, f)
    T = len(f["C"])
    pres = f["present"] > 0
    hp = 1 - np.prod(1 - 1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    hits = TR.hits_from(hp, nms=8) if TR.ARCH == "rallyphys" else TR.hits_from(hp)
    who = np.argmax(1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
    bt_f = np.array(g["back_track"]["frames"], float)
    bt_px = np.array(g["back_track"]["px"], float).reshape(-1, 2)
    f0, f1 = max(0, int(bt_f[0]) - PRE), min(T - 1, int(bt_f[-1]) + POST)
    frames = np.arange(f0, f1 + 1)
    rk_path = os.path.join(ROOT, "outputs", "racket", m, r + ".npz")
    boxes_all = load_rackets(rk_path) if os.path.exists(rk_path) else []
    det = {int(round(a)): uv for a, uv in zip(bt_f, bt_px)}
    if TR.PHYS:
        S = TR.physics_shuttle(p, f, out)
    elif TR.ARCH == "rallyphys":  # as scored (tools/train_rallynet.evaluate_one)
        S = TR.rallyphys_shuttle(p, f, out)
    else:
        S = out["shuttle"]
    Jt = out["joints"] if TR.ARCH == "rallyphys" else RN.fix_head(out["joints"])
    # racket: the player's one racket hand for the rally + model direction
    if TR.ARCH == "rallyphys":  # the network's hand, one per player for the whole rally
        hand = (np.nanmean(np.where(pres, out["hand_p"], np.nan), 0) > 0.5).astype(int)
        u = out["u2"][:, np.arange(RN.P), hand]
        u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    else:
        hand = RN.racket_hand(out["joints"], f)
        u = out["racket_u"]
    wrist = Jt[:, np.arange(Jt.shape[1]), 9 + hand]
    gt_by = {}
    for t, *X in gt["shuttle"]:
        k = int(round(t))
        if abs(t - k) <= 0.5:
            gt_by[k] = X
    per = []
    normals = [face_normals(wrist[frames, q], u[frames, q]) for q in range(RN.P)]
    for i, fr in enumerate(frames):
        P, Rk = [], []
        for q in range(RN.P):
            if not pres[fr, q]:
                continue
            X = Jt[fr, q]
            P.append({"id": q, "s": "n" if X[[15, 16], 1].mean() < NET_Y else "f", "j": [r3(x) for x in X],
                      "k": [norm_px(x, cam.size) for x in cam.project(X)]})
            w_, u_ = wrist[fr, q], u[fr, q]
            butt, tip = w_ - GRIP * u_, w_ + (RACKET_LEN - GRIP) * u_
            nv = normals[q][i]
            Rk.append({"id": q, "w": r3(w_), "u": r3(u_), "seen": True, **({"n": r3(nv)} if np.isfinite(nv).all() else {}),
                       "px": [norm_px(x, cam.size) for x in cam.project(np.array([butt, tip]))]})
        bx = boxes_all[fr] if fr < len(boxes_all) else np.zeros((0, 5))
        per.append({"p": P, "r": Rk, "rb": [norm_px(b[:2], cam.size) + norm_px(b[2:4], cam.size) for b in bx if b[4] >= 0.25],
                    "x": r3(S[fr]), "xp": norm_px(cam.project(S[fr][None])[0], cam.size),
                    "d": norm_px(det[fr], cam.size) if fr in det else None,
                    "g": r3(np.array(gt_by[fr])) if fr in gt_by else None,
                    "pg": [[round(a, 2), round(b, 2)] for a, b in gt["players"].get(str(fr), [])]})
    ck = lambda t: round(float(t) - f0, 2)
    shots = []
    hs = [h for h in hits if f0 <= h <= f1]
    for k, h in enumerate(hs):
        e = hs[k + 1] if k + 1 < len(hs) else min(f1, h + 60)
        seg = S[h:e + 1]
        v = np.linalg.norm(np.diff(seg[:6], axis=0), axis=1) * float(1 / f["dt"]) if len(seg) > 2 else [0]
        q = who[h]
        shots.append({"t0": ck(h), "t1": ck(e), "side": "near" if Jt[h, q, [15, 16], 1].mean() < NET_Y else "far",
                      "speed": float(np.max(v) * 3.6), "apex": float(seg[:, 2].max()), "anchored": True,
                      "path": [r3(x) for x in seg]})
    ev = TR.evaluate_one(model, p, out=out, S=S)
    R = cam.R
    clip = f"{m}_{r}.mp4"
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(OUT, clip)):
        write_clip(os.path.join(ROOT, "data", "videos", m, r, "(1).mp4"), os.path.join(OUT, clip), f0, f1)
    meta = {"id": f"{m} {r}", "clip": clip, "skel": "coco17", "n": len(frames), "f0": int(f0),
            "cam": {"c": r3(cam.center), "fwd": r3(R.T @ [0, 0, 1.0]), "down": r3(R.T @ [0, 1.0, 0]),
                    "vfov": round(float(np.degrees(2 * np.arctan(cam.size[1] / 2 / cam.f))), 2),
                    "aspect": round(cam.size[0] / cam.size[1], 4)},
            "shots": shots, "labels": [ck(l) for l in gt["hits"] if f0 <= l <= f1],
            "gt_path": [[ck(t)] + [round(float(x), 3) for x in X] for t, *X in gt["shuttle"]],
            "racket": {"len": RACKET_LEN, "head": list(RACKET_HEAD), "grip": GRIP},
            "metrics": {"player_err_median": ev["player"], "err_median": ev["shuttle"], "err_mean": ev["shuttle_mean"],
                        "n_gt": ev["n_gt"], "hits_detected": len(hs), "hits_labeled": len(gt["hits"]),
                        "hit_recall": ev["recall"], "has_racket": True},
            "frames": per}
    print(mr, meta["metrics"], flush=True)
    return meta


if __name__ == "__main__":
    model = TR.load_model(TAG)
    data = [export(mr, model) for mr in sys.argv[1:]]
    json.dump(data, open(os.path.join(OUT, "data.json"), "w"), separators=(",", ":"))
    print("wrote", os.path.join(OUT, "data.json"), os.path.getsize(os.path.join(OUT, "data.json")) // 1024, "KB")
