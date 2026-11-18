"""Fine-tune Birdseye v1's hit + stroke transformer (3:Action, HitSpotterTransformer) on FineBadminton-20K
broadcast rallies: VIMO skeletons (tools/fb_vimo.py), FineBadminton hit frames and FBBench's stroke names.

Same input, targets and recipe as v1 (train_final.py / build_training_data_selfdetected.py): SMPL-24
pelvis-relative joints, top player rotated 180 deg, velocity, acceleration, opponent context, 30 fps,
triangular hit target +-1/3 s, stroke label at the hit frame, loss = hit BCE + 0.3 stroke CE, mirror
augmentation. Starts from v1's checkpoint; the stroke head's last layer is replaced (new label set).

Split by match: TEST_GROUPS = test (never used for training or selection); VAL_MATCHES pick the epoch
and the peak threshold; the rest train. FineBadminton-20K never includes FBBench's source matches.
usage: python tools/v1_finetune.py   -> outputs/v1ft/model.pt, outputs/v1ft/report.json"""
import glob, json, os, random, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2 import v1action as A
from birdseye_v2.broadcast import FPS, native_frames, native_index
from birdseye_v2.fbfeat import STROKE_NAMES

CLIPS = os.path.join(ROOT, "data", "fb20k", "clips")
VIMO = os.path.join(ROOT, "outputs", "fb20k", os.environ.get("VIMO_DIR", "vimo_bc"))
OUT = os.path.join(ROOT, "outputs", os.environ.get("V1FT_OUT", "v1ft"))
TEST_GROUPS = {"uncalibrated": {"0011", "0016"},  # court calibration fails on every rally: nominal camera
               "calibrated": {"0020", "0068"}}   # calibration trusted on every rally
TEST_MATCHES = set().union(*TEST_GROUPS.values())
VAL_MATCHES = {"0055", "0065"}  # not FBBench sources (0001-0010, 0041-0051 are)
CLASSES = sorted(set(STROKE_NAMES.values()))  # 11 incl. Serve
WIN = 30.0 / 3.0  # frames at 30 fps (v1: WINDOW_SEC = 1/3)
DEV = "cuda"


def load_rally(uid):
    lab = json.load(open(os.path.join(CLIPS, uid + ".json"), encoding="utf8"))
    v = np.load(os.path.join(VIMO, uid + ".npz"))
    top, bot = v["top"], v["bottom"]
    if np.isnan(top[:, 0, 0]).mean() > 0.5 or np.isnan(bot[:, 0, 0]).mean() > 0.5:
        return None
    keep = native_frames(os.path.join(CLIPS, uid + ".mp4"), os.path.join(ROOT, "outputs", "fb20k", "native", uid + ".json"))
    jt, t30 = A._resample(A._interp_nan(top), FPS, 30.0)
    jb, _ = A._resample(A._interp_nan(bot), FPS, 30.0)
    rel = {"top": A._canon(jt, "top"), "bottom": A._canon(jb, "bottom")}
    n = len(t30)
    tgt = {s: np.zeros(n, np.float32) for s in rel}
    shot = {s: np.full(n, -1, np.int64) for s in rel}
    hits = []
    for h in lab["hits"]:
        s = h["hitter"]
        f30 = native_index(keep, h["frame"]) / FPS * 30.0
        lo, hi = max(0, int(f30 - WIN)), min(n, int(f30 + WIN) + 1)
        for f in range(lo, hi):
            tgt[s][f] = max(tgt[s][f], max(0.0, 1 - abs(f - f30) / WIN))
        fi = int(round(f30))
        if 0 <= fi < n:
            shot[s][fi] = CLASSES.index(STROKE_NAMES.get(h["type"], h["type"])) if STROKE_NAMES.get(h["type"], h["type"]) in CLASSES else -1
        hits.append((f30, s, STROKE_NAMES.get(h["type"], h["type"])))
    return {"uid": uid, "match": uid[:4], "rel": rel, "tgt": tgt, "shot": shot, "hits": hits, "n": n}


def samples(rallies):
    out = []
    for r in rallies:
        for s, o in (("top", "bottom"), ("bottom", "top")):
            out.append((r["rel"][s], r["rel"][o], r["tgt"][s], r["shot"][s]))
    return out


def batchify(items, augment):
    feats, tg, sh = [], [], []
    for sr, orl, t, s in items:
        if augment and random.random() < 0.5:
            sr, orl = A._mirror(sr), A._mirror(orl)
        feats.append(A._feats(sr, orl)); tg.append(t); sh.append(s)
    L = max(len(f) for f in feats)
    X = np.zeros((len(feats), L, 360), np.float32); Tg = np.zeros((len(feats), L), np.float32)
    Sh = np.full((len(feats), L), -1, np.int64); M = np.zeros((len(feats), L), bool)
    for i, (f, t, s) in enumerate(zip(feats, tg, sh)):
        X[i, :len(f)] = f; Tg[i, :len(f)] = t; Sh[i, :len(f)] = s; M[i, :len(f)] = True
    return [torch.tensor(a).to(DEV) for a in (X, Tg, Sh, M)]


def predict(model, r):
    out = {}
    for s, o in (("top", "bottom"), ("bottom", "top")):
        x = torch.tensor(A._feats(r["rel"][s], r["rel"][o]))[None].to(DEV)
        xm = torch.tensor(A._feats(A._mirror(r["rel"][s]), A._mirror(r["rel"][o])))[None].to(DEV)
        with torch.no_grad():
            hl, sl = model(x); hlm, slm = model(xm)
        out[s] = (((torch.sigmoid(hl) + torch.sigmoid(hlm)) / 2)[0].cpu().numpy(),
                  ((torch.softmax(sl, -1) + torch.softmax(slm, -1)) / 2)[0].cpu().numpy())
    return out


def evaluate(model, rallies, thr, tols=(3, 8)):
    """tol in 25 fps frames, as tools/hit_eval_fb.py."""
    res = {t: {"tp": 0, "fp": 0, "fn": 0, "side": 0, "type": 0} for t in tols}
    model.eval()
    for r in rallies:
        pr = predict(model, r)
        pred = []
        for s in ("top", "bottom"):
            hp, sp = pr[s]
            peaks = []
            for i in np.argsort(-hp):
                if hp[i] < thr:
                    break
                if all(abs(i - p) >= 10 for p in peaks):
                    peaks.append(int(i))
            pred += [(p * FPS / 30.0, s, CLASSES[int(sp[p].argmax())]) for p in peaks]
        gt = [(f * FPS / 30.0, s, t) for f, s, t in r["hits"]]
        for tol in tols:
            pairs = sorted((abs(p[0] - g[0]), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p[0] - g[0]) <= tol)
            up, ug = set(), set()
            for _, i, j in pairs:
                if i not in up and j not in ug:
                    up.add(i); ug.add(j)
                    res[tol]["side"] += pred[i][1] == gt[j][1]
                    res[tol]["type"] += pred[i][2] == gt[j][2]
            res[tol]["tp"] += len(up); res[tol]["fp"] += len(pred) - len(up); res[tol]["fn"] += len(gt) - len(up)
    out = {}
    for tol, d in res.items():
        P = d["tp"] / max(d["tp"] + d["fp"], 1); R = d["tp"] / max(d["tp"] + d["fn"], 1)
        out[tol] = {"P": round(P, 3), "R": round(R, 3), "F1": round(2 * P * R / max(P + R, 1e-9), 3),
                    "side": round(d["side"] / max(d["tp"], 1), 3), "type": round(d["type"] / max(d["tp"], 1), 3)}
    return out


def main():
    random.seed(0); torch.manual_seed(0)
    uids = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(VIMO, "*.npz")))
    rallies = [r for r in (load_rally(u) for u in uids) if r]
    test = [r for r in rallies if r["match"] in TEST_MATCHES]
    val = [r for r in rallies if r["match"] in VAL_MATCHES]
    train = [r for r in rallies if r["match"] not in TEST_MATCHES | VAL_MATCHES]
    if os.environ.get("TRAIN_UIDS"):  # data-size ablation: train on a listed subset only
        keep = {l.strip() for l in open(os.environ["TRAIN_UIDS"]) if l.strip()}
        train = [r for r in train if r["uid"] in keep]
    print(f"rallies train {len(train)} val {len(val)} test {len(test)}  classes {CLASSES}", flush=True)
    model = A.load(DEV)
    model.shot_head[-1] = nn.Linear(model.shot_head[-1].in_features, len(CLASSES)).to(DEV)
    tr = samples(train)
    pos = np.concatenate([t for _, _, t, _ in tr]); pos_w = min(10.0, (1 - (pos > 0.5).mean()) / max((pos > 0.5).mean(), 1e-4))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    E = int(os.environ.get("EPOCHS", 30))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=E)
    best, best_state, best_thr = -1, None, 0.5
    for ep in range(E):
        model.train(); random.shuffle(tr); t0 = time.time(); tot = 0
        for b in range(0, len(tr), 4):
            X, Tg, Sh, M = batchify(tr[b:b + 4], augment=True)
            hl, sl = model(X, pad_mask=~M)
            w = 1.0 + (pos_w - 1.0) * Tg
            hit = ((F.binary_cross_entropy_with_logits(hl, Tg, reduction="none") * w) * M).sum() / M.sum()
            sm = (Sh >= 0) & M
            sl_loss = F.cross_entropy(sl[sm], Sh[sm]) if sm.any() else torch.tensor(0.0, device=DEV)
            loss = hit + 0.3 * sl_loss
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        sched.step()
        if ep % 3 == 2 or ep == E - 1:
            for thr in (0.3, 0.4, 0.5, 0.6):
                v = evaluate(model, val, thr)[8]
                if v["F1"] > best:
                    best, best_thr = v["F1"], thr
                    best_state = {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}
            diag = " ".join(f"{g} {te[8]['F1']:.3f}/{te[3]['F1']:.3f}/{te[8]['type']:.3f}"  # diagnostic only, not for selection
                            for g, te in ((g, evaluate(model, [r for r in test if r["match"] in ms], best_thr))
                                          for g, ms in TEST_GROUPS.items()))
            print(f"ep {ep} loss {tot / max(len(tr) // 4, 1):.4f} val@8 best F1 {best:.3f} thr {best_thr} "
                  f"| test F1@8/F1@3/type {diag} ({time.time() - t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    os.makedirs(OUT, exist_ok=True)
    torch.save({"state": best_state, "classes": CLASSES, "thr": best_thr}, os.path.join(OUT, "model.pt"))
    rep = {"val_F1@8": best, "thr": best_thr, "test": {g: evaluate(model, [r for r in test if r["match"] in ms], best_thr) for g, ms in TEST_GROUPS.items()},
           "n": {"train": len(train), "val": len(val), "test": len(test)}}
    print(json.dumps(rep, indent=1))
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
