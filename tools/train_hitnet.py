"""Train HitNet (birdseye_v2/hitnet.py) on the hit labels of BFMD / FineBadminton (who hits when) and the DroneData
training matches (when); evaluate on the DroneData test matches: labelled hits recovered within 3 frames (recall),
found hits next to a label (precision), and the hitter's court side.

usage: python tools/train_hitnet.py [train|eval]   (env STEPS=30000, CROP=192, BATCH=24, HTAG=h0, DRONE_TRAIN=...)"""
import glob, json, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
for k in ("VTEACH", "CONTACT3D", "SHUTTLE3D"):  # RallyPhys teachers: not needed here
    os.environ[k] = "0"
os.environ.setdefault("CROP", "192")  # training crops (= HCROP)
import train_rallynet as TR  # noqa: E402
from birdseye_v2 import hitnet as HN  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "hitnet")
HTAG = os.environ.get("HTAG", "h0")
HCROP = 192  # window (frames) HitNet is trained and run on
STEPS = int(os.environ.get("STEPS", "30000"))
BATCH = int(os.environ.get("BATCH", "24"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "5000"))
DEV = TR.DEV


def load(tag=None):
    m = HN.HitNet().to(DEV)
    m.load_state_dict(torch.load(os.path.join(OUT, (tag or HTAG) + ".pt"), map_location=DEV))
    return m.eval()


@torch.no_grad()
def infer(model, f, win=None, stride=None):
    """Whole-rally hit logits (T,P) by overlapping windows, averaged with a triangular weight."""
    win = win or HCROP
    stride = stride or win // 2
    model.eval()
    T = len(f["C"])
    acc, ws = np.zeros((T, TR.RN.P)), np.zeros(T)
    starts = list(range(0, max(1, T - win + 1), stride))
    if starts[-1] + win < T:
        starts.append(T - win)
    for a in starts:
        b = TR.to_dev({k: v[None] for k, v in TR.crop(f, a, a + win).items()})
        lo = model(b)[0].cpu().numpy()
        n = min(win, T - a)
        w = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)).astype(float)
        acc[a:a + n] += lo[:n] * w[:, None]
        ws[a:a + n] += w
    return acc / ws[:, None]


def score(model, paths, thr=0.5, verbose=False):
    from birdseye_v2.court import NET_Y
    rec, prec, side = [], [], []
    for p in paths:
        f = TR.load_feats(p)
        gt = json.load(open(p[:-4] + "_gt.json"))
        sp = 1 / (1 + np.exp(-infer(model, f))) * f["present"]
        hp = 1 - np.prod(1 - sp, axis=1)
        hits = TR.hits_from(hp, thr=thr, nms=8)
        L = np.array(gt["hits"], float)
        if len(L):
            rec += [np.any(np.abs(hits - l) <= 3) for l in L]
        prec += [len(L) > 0 and np.min(np.abs(L - h)) <= 3 for h in hits]
        # the hitter's side: the most likely player at the hit vs the GT shuttle's side just after it
        G = np.array(gt["shuttle"], float).reshape(-1, 4)
        for h in hits:
            g = G[(G[:, 0] >= h - 1) & (G[:, 0] <= h + 2)]
            if len(g):
                q = int(np.argmax(sp[int(h)]))
                side.append((f["root"][int(h), q, 1] > NET_Y) == (g[0, 2] > NET_Y))
    r, pr = np.mean(rec), np.mean(prec)
    return {"recall": round(float(r), 3), "precision": round(float(pr), 3), "f1": round(float(2 * r * pr / (r + pr)), 3),
            "side": round(float(np.mean(side)), 3), "labels": len(rec), "found": len(prec)}


def train():
    test = set(TR.labels()["bfmd_test"])
    paths = [p for s in TR.SETS for p in glob.glob(os.path.join(TR.OBS, s, "*.npz")) if os.path.basename(p)[:-4] not in test]
    paths += [p for p in glob.glob(os.path.join(TR.OBS, "drone", "*.npz")) if os.path.basename(p).startswith(TR.TRAIN_DRONE)] * 3
    print(len(paths), "training rallies", flush=True)
    model = HN.HitNet().to(DEV)
    print(sum(p.numel() for p in model.parameters()) / 1e6, "M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
    dl = DataLoader(TR.Crops(paths, STEPS * BATCH), batch_size=BATCH, num_workers=int(os.environ.get("NW", "6")), persistent_workers=True)
    os.makedirs(OUT, exist_ok=True)
    check = TR.eval_paths()  # the 60 check rallies (test matches)
    t0, run = time.time(), {}
    for step, b in enumerate(dl):
        b = TR.to_dev(b)
        lo = model(b)
        l, parts = HN.loss(lo, b)
        if not torch.isfinite(l):
            print("skip: non-finite loss", flush=True)
            continue
        opt.zero_grad()
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        for k, v in parts.items():
            run[k] = 0.98 * run.get(k, v) + 0.02 * v
        if step % 200 == 0:
            print(f"{step} {time.time() - t0:.0f}s " + " ".join(f"{k} {v:.3g}" for k, v in run.items()), flush=True)
        if step % 2000 == 0 or step == STEPS - 1:
            torch.save(model.state_dict(), os.path.join(OUT, HTAG + ".pt"))
        if step and (step % EVAL_EVERY == 0 or step == STEPS - 1):
            print("EVAL", step, score(model, check), flush=True)
            model.train()
        if step >= STEPS - 1:
            break


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train()
    else:
        m = load()
        full = TR.eval_paths(full=True)
        for thr in (0.3, 0.4, 0.5, 0.6):
            print("thr", thr, score(m, full, thr=thr), flush=True)
