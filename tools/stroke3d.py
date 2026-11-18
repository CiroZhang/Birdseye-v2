"""Hit-centred stroke classifier over 3D features, on the eval of tools/bfmd_stroke.py (PROTOCOL=clean): same KEEP
rallies, v1 rally filter, test / selection / training matches, per-frame features (v1's make_features of the chosen
representation, with FEAT points appended exactly as bfmd_stroke.py does) and metrics at the labelled hit frames.

Unlike v1's model (whole rally, hit + stroke heads, stroke label on the hit frame only) this classifies one window of
2W+1 frames centred on each labelled hit, stroke only, with the recipe of the RGB baseline (tools/bfmd_rgb_stroke.py):
+-2 frame centre jitter, left-right mirroring, label smoothing 0.1, AdamW + warmup/cosine. Model: frame tokens
(linear + LayerNorm) + learned positions + a CLS token -> pre-norm transformer encoder -> CLS and centre token -> class.

  SWEEP=1  train every config in CONFIGS on the training matches, report its best selection macro-F1 (no test)
  else     train CFG (json, or the name of a config) for SEEDS, epoch picked on selection macro-F1, test scored once;
           REFIT=1 then retrains on training + selection matches for the picked number of epochs

INPUT=rgb: the same model and recipe on video instead: each frame (tools/bfmd_frames.py, SIZE x SIZE) becomes a token
through a small CNN trained from scratch (no pretraining), plus a learned hitter (top/bottom) embedding on the CLS token;
mirroring = horizontal flip. INPUT=3d (default) is unchanged.

env as bfmd_stroke.py for the data (BFMD_DATA_ROOT, FEAT, SD, KEEP); out outputs/bfmd/stroke3d/<TAG>.json"""
import copy, json, os, sys, time

os.environ.setdefault("PROTOCOL", "clean")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import bfmd_stroke as BS  # noqa: E402  (sets up train_final as BS.tf and the feature helpers)

tf = BS.tf
OUT = os.path.join(ROOT, "outputs", "bfmd", "stroke3d")
CONFIGS = {
    "w16_d128_l2": dict(W=16, d=128, L=2, drop=0.1),
    "w16_d192_l4": dict(W=16, d=192, L=4, drop=0.1),
    "w24_d192_l4": dict(W=24, d=192, L=4, drop=0.1),
    "w24_d192_l4_dr3": dict(W=24, d=192, L=4, drop=0.3),
    "w32_d192_l4": dict(W=32, d=192, L=4, drop=0.1),
    "w32_d256_l6_dr3": dict(W=32, d=256, L=6, drop=0.3),
    "w48_d192_l4_dr3": dict(W=48, d=192, L=4, drop=0.3),
}
INPUT = os.environ.get("INPUT", "3d")
FRAMES = os.environ.get("FRAMES_DIR") or os.path.join(ROOT, "data", "bfmd", "frames" + os.environ.get("SIZE", "128"))
DEFAULT = dict(epochs=60, bs=64, lr=5e-4, wd=0.05, ls=0.1, jitter=2, heads=4)


def load():
    """Per-sample (features, mirrored features, stroke labels per frame) for train / selection / test, as bfmd_stroke."""
    dataset = json.load(open(f"{tf.BASE}/training_dataset_all_selfdetected.json"))
    if os.path.normpath(tf.BASE).split(os.sep)[-2:-1] == ["fbs"]:  # FineBadminton-20K inputs, as stroke_baselines.py
        BS.SRC.update(out=os.path.join(ROOT, "outputs", "fb20k"),
                      uid=lambda m, rt: f"{m}_{int(rt.split('_r')[1]):03d}")
    if BS.FEATS & {"shuttle", "shuttle2d", "racket", "own"}:
        BS.set_pelvis(dataset)
    keep = set(open(BS.KEEP).read().split())
    dataset["rallies"] = {t: r for t, r in dataset["rallies"].items() if t in keep}
    if BS.PAIR:
        BS.add_pair(dataset)
    if "shuttle" in BS.FEATS:
        BS.add_shuttle(dataset)
    if "shuttle2d" in BS.FEATS:
        BS.add_shuttle2d(dataset)
    if "own" in BS.FEATS:
        BS.add_own(dataset)
    if "racket" in BS.FEATS:
        BS.add_racket(dataset)
        tf.LR_PERM = tf.LR_PERM + [len(tf.LR_PERM)]
    vocab = dataset["shot_type_vocab"]
    s2i = {s: i for i, s in enumerate(vocab)}
    _, test_tags, test_m = tf.load_split(dataset)
    mor = dataset["match_of_rally"]
    tr_m, sel_m = BS.split_matches(mor, set(mor.values()) - test_m)
    tags = sorted(dataset["rallies"])
    sets = {}
    for name, tg in (("train", [t for t in tags if mor[t] in tr_m]), ("sel", [t for t in tags if mor[t] in sel_m]),
                     ("test", test_tags)):
        ds = tf.RallyDataset(dataset, tg, s2i, augment=False)
        items = []
        for s in ds.samples:
            sr, sv, sa, orl, ov = s[:5]
            feat = tf.make_features(sr, sv, sa, orl, ov)
            mir = tf.make_features(tf.mirror_lr(sr), tf.mirror_lr(sv), tf.mirror_lr(sa), tf.mirror_lr(orl), tf.mirror_lr(ov))
            items.append((np.nan_to_num(feat).astype(np.float32), np.nan_to_num(mir).astype(np.float32), s[6]))
        sets[name] = items
    del dataset
    return sets, len(vocab)


def load_rgb():
    """Per side of each kept rally: (frame offset, n frames, stroke labels per frame, side 0=top/1=bottom)."""
    dataset = json.load(open(f"{tf.BASE}/training_dataset_all_selfdetected.json"))
    keep = set(open(BS.KEEP).read().split())
    dataset["rallies"] = {t: r for t, r in dataset["rallies"].items() if t in keep}
    idx = json.load(open(os.path.join(FRAMES, "index.json")))
    vocab = dataset["shot_type_vocab"]
    s2i = {s: i for i, s in enumerate(vocab)}
    _, test_tags, test_m = tf.load_split(dataset)
    mor = dataset["match_of_rally"]
    tr_m, sel_m = BS.split_matches(mor, set(mor.values()) - test_m)
    tags = sorted(dataset["rallies"])
    sets = {}
    for name, tg in (("train", [t for t in tags if mor[t] in tr_m]), ("sel", [t for t in tags if mor[t] in sel_m]),
                     ("test", sorted(test_tags))):
        items = []
        for t in tg:
            r = dataset["rallies"][t]
            if r["nan_frac_top"] > 0.5 or r["nan_frac_bottom"] > 0.5:  # v1's rally filter, as RallyDataset
                continue
            off, n = idx["index"][t]
            for side, key in ((0, "shot_type_top"), (1, "shot_type_bottom")):
                shot = np.array([s2i.get(x, -1) for x in r[key]], np.int64)
                items.append((off, min(n, len(shot)), shot, side))
        sets[name] = items
    mm = np.memmap(os.path.join(FRAMES, "frames.u8"), np.uint8, "r", shape=(idx["n"], idx["size"], idx["size"], 3))
    return sets, len(vocab), mm


def windows_rgb(items, W, jitter=0, mirror_p=0.0, rng=None):
    """(N, 2W+1) global frame indices (-1 = outside the rally), (N, 2W+1) validity, (N,) labels, (N,) side, (N,) flip."""
    I, M, Y, S, Fl = [], [], [], [], []
    for off, n, shot, side in items:
        for f in np.flatnonzero(shot[:n] >= 0):
            c = f + (rng.integers(-jitter, jitter + 1) if jitter else 0)
            loc = np.arange(c - W, c + W + 1)
            ok = (loc >= 0) & (loc < n)
            I.append(np.where(ok, off + loc, -1)); M.append(ok); Y.append(shot[f]); S.append(side)
            Fl.append(bool(mirror_p and rng.random() < mirror_p))
    return np.stack(I), np.stack(M), np.array(Y, np.int64), np.array(S, np.int64), np.array(Fl)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255


def gather(mm, I, Fl, dev):
    """(B, 2W+1) frame indices -> normalised (B, 2W+1, 3, H, W) on dev; invalid frames are zeros."""
    B, L = I.shape
    flat = I.reshape(-1)
    x = np.zeros((len(flat),) + mm.shape[1:], np.uint8)
    ok = flat >= 0
    order = np.argsort(flat[ok])  # sorted reads from the memmap
    x[np.flatnonzero(ok)[order]] = mm[flat[ok][order]]
    x = torch.from_numpy(x).to(dev, non_blocking=True).permute(0, 3, 1, 2).float()
    x = ((x - MEAN.to(dev)) / STD.to(dev)) * torch.from_numpy(ok).to(dev).view(-1, 1, 1, 1)
    x = x.view(B, L, *x.shape[1:])
    fl = torch.from_numpy(Fl).to(dev)
    return torch.where(fl.view(B, 1, 1, 1, 1), x.flip(-1), x)


class FrameCNN(nn.Module):
    """Small per-frame encoder trained from scratch: 5 stride-2 conv stages -> global pool -> d."""
    def __init__(self, d):
        super().__init__()
        ch, layers, c0 = (32, 64, 128, 192, 256), [], 3
        for c in ch:
            layers += [nn.Conv2d(c0, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.GELU(),
                       nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.GELU()]
            c0 = c
        self.net = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Linear(c0, d), nn.LayerNorm(d))

    def forward(self, x):  # (B, L, 3, H, W) -> (B, L, d)
        B, L = x.shape[:2]
        return self.out(self.net(x.flatten(0, 1)).mean((2, 3))).view(B, L, -1)


def windows(items, W, jitter=0, mirror_p=0.0, rng=None):
    """(N, 2W+1, D) windows centred on every labelled hit (+-jitter), (N, 2W+1) validity, (N,) labels."""
    X, M, Y = [], [], []
    for feat, mir, shot in items:
        T = len(feat)
        for f in np.flatnonzero(shot >= 0):
            c = f + (rng.integers(-jitter, jitter + 1) if jitter else 0)
            idx = np.arange(c - W, c + W + 1)
            ok = (idx >= 0) & (idx < T)
            src = mir if (mirror_p and rng.random() < mirror_p) else feat
            x = np.zeros((2 * W + 1, feat.shape[1]), np.float32)
            x[ok] = src[idx[ok]]
            X.append(x); M.append(ok); Y.append(shot[f])
    return np.stack(X), np.stack(M), np.array(Y, np.int64)


class Net(nn.Module):
    def __init__(self, D, n_cls, W, d, L, drop, heads):
        super().__init__()
        self.rgb = D is None
        self.inp = FrameCNN(d) if self.rgb else nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
        self.side = nn.Embedding(2, d) if self.rgb else None
        self.pos = nn.Parameter(torch.zeros(1, 2 * W + 2, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, L)
        self.norm = nn.LayerNorm(2 * d)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(2 * d, n_cls)
        self.W = W

    def forward(self, x, m, side=None):
        cls = self.cls.expand(len(x), -1, -1)
        if self.rgb:
            cls = cls + self.side(side)[:, None]
        h = torch.cat([cls, self.inp(x)], 1) + self.pos
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), ~m], 1)
        h = self.enc(h, src_key_padding_mask=pad)
        return self.head(self.drop(self.norm(torch.cat([h[:, 0], h[:, 1 + self.W]], 1))))


def metrics(P, Y):
    f1 = []
    for c in np.unique(Y):
        tp, fp, fn = ((P == c) & (Y == c)).sum(), ((P == c) & (Y != c)).sum(), ((P != c) & (Y == c)).sum()
        f1.append(2 * tp / (2 * tp + fp + fn))
    return float((P == Y).mean()), float(np.mean(f1))


def predict(model, X, M, dev):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            out.append(model(torch.from_numpy(X[i:i + 512]).to(dev), torch.from_numpy(M[i:i + 512]).to(dev)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def train(cfg, train_items, n_cls, seed, dev, sel=None, epochs=None):
    """Returns (model at the best selection epoch or the last epoch, best epoch, log)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    epochs = epochs or cfg["epochs"]
    D = train_items[0][0].shape[1]
    model = Net(D, n_cls, cfg["W"], cfg["d"], cfg["L"], cfg["drop"], cfg["heads"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n_hits = sum(int((s >= 0).sum()) for _, _, s in train_items)
    steps = epochs * (n_hits // cfg["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(steps, 1), pct_start=0.1)
    if sel is not None:
        Xs, Ms, Ys = windows(sel, cfg["W"])
    best, best_ep, best_state, log = -1, -1, None, []
    for ep in range(epochs):
        model.train()
        X, M, Y = windows(train_items, cfg["W"], cfg["jitter"], 0.5, rng)
        perm = rng.permutation(len(X))
        tot = 0
        for b in range(len(X) // cfg["bs"]):
            i = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            x, m, y = (torch.from_numpy(a[i]).to(dev) for a in (X, M, Y))
            loss = F.cross_entropy(model(x, m), y, label_smoothing=cfg["ls"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        if sel is not None:
            acc, f1 = metrics(predict(model, Xs, Ms, dev), Ys)
            log.append({"epoch": ep, "loss": tot / max(1, len(X) // cfg["bs"]), "sel_acc": acc, "sel_f1": f1})
            if f1 > best:
                best, best_ep, best_state = f1, ep, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, log


def predict_rgb(model, mm, I, M, S, dev, bs=32):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(I), bs):
            x = gather(mm, I[i:i + bs], np.zeros(len(I[i:i + bs]), bool), dev)
            out.append(model(x, torch.from_numpy(M[i:i + bs]).to(dev), torch.from_numpy(S[i:i + bs]).to(dev)).argmax(1).float().cpu().numpy())
    return np.concatenate(out).astype(np.int64)


def train_rgb(cfg, train_items, n_cls, seed, dev, mm, sel=None, epochs=None):
    """train() for INPUT=rgb: identical schedule, loss, jitter, mirroring and epoch selection."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    epochs = epochs or cfg["epochs"]
    model = Net(None, n_cls, cfg["W"], cfg["d"], cfg["L"], cfg["drop"], cfg["heads"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    n_hits = sum(int((s[:n] >= 0).sum()) for _, n, s, _ in train_items)
    steps = epochs * (n_hits // cfg["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(steps, 1), pct_start=0.1)
    if sel is not None:
        Is, Ms, Ys, Ss, _ = windows_rgb(sel, cfg["W"])
    best, best_ep, best_state, log = -1, -1, None, []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        I, M, Y, S, Fl = windows_rgb(train_items, cfg["W"], cfg["jitter"], 0.5, rng)
        perm = rng.permutation(len(I))
        tot, nb = 0, len(I) // cfg["bs"]
        for b in range(nb):
            i = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            x = gather(mm, I[i], Fl[i], dev)
            m, y, sd = (torch.from_numpy(a[i]).to(dev) for a in (M, Y, S))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x, m, sd), y, label_smoothing=cfg["ls"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        msg = f"epoch {ep:2d}: loss {tot / max(nb, 1):.3f} ({time.time() - t0:.0f}s)"
        if sel is not None:
            acc, f1 = metrics(predict_rgb(model, mm, Is, Ms, Ss, dev), Ys)
            log.append({"epoch": ep, "loss": tot / max(nb, 1), "sel_acc": acc, "sel_f1": f1})
            msg += f" | sel acc={acc:.3f} macroF1={f1:.3f}"
            if f1 > best:
                best, best_ep, best_state = f1, ep, copy.deepcopy(model.state_dict())
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, log


def main_rgb(dev):
    sets, n_cls, mm = load_rgb()
    if os.environ.get("LIMIT"):  # smoke tests only
        sets = {k: v[:int(os.environ["LIMIT"])] for k, v in sets.items()}
    hits = lambda it: sum(int((s[:n] >= 0).sum()) for _, n, s, _ in it)
    print(f"rgb: hits train {hits(sets['train'])} sel {hits(sets['sel'])} test {hits(sets['test'])}", flush=True)
    tag = os.environ.get("TAG", "stroke_rgbscratch")
    c = os.environ.get("CFG", "w32_d192_l4")
    cfg = {**DEFAULT, **(CONFIGS[c] if c in CONFIGS else json.loads(c))}
    cfg["bs"] = int(os.environ.get("BS", "32"))
    refit = os.environ.get("REFIT") == "1"
    It, Mt, Yt, St, _ = windows_rgb(sets["test"], cfg["W"])
    runs = []
    for seed in [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]:
        model, ep, log = train_rgb(cfg, sets["train"], n_cls, seed, dev, mm, sets["sel"])
        if refit:
            model, _, _ = train_rgb(cfg, sets["train"] + sets["sel"], n_cls, seed, dev, mm, None, epochs=ep + 1)
        acc, f1 = metrics(predict_rgb(model, mm, It, Mt, St, dev), Yt)
        runs.append({"seed": seed, "epoch": ep, "test_acc": acc, "test_macro_f1": f1, "sel": log[ep]})
        print(f"RESULT {tag} seed {seed}: epoch {ep} sel macroF1={log[ep]['sel_f1']:.3f} | test at true hits acc={acc:.3f} macroF1={f1:.3f} (n={len(Yt)})", flush=True)
    A, Fm = np.array([r["test_acc"] for r in runs]), np.array([r["test_macro_f1"] for r in runs])
    sd = lambda a: 100 * a.std(ddof=1) if len(a) > 1 else 0
    print(f"MEAN {tag}: acc {100 * A.mean():.1f} +- {sd(A):.1f}, macroF1 {100 * Fm.mean():.1f} +- {sd(Fm):.1f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"tag": tag, "input": "rgb", "cfg": cfg, "refit": refit, "frames": FRAMES, "runs": runs},
              open(os.path.join(OUT, tag + ".json"), "w"), indent=1)


def main():
    if INPUT == "rgb":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = False
        return main_rgb(torch.device("cuda"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    t0 = time.time()
    sets, n_cls = load()
    print(f"loaded in {time.time() - t0:.0f}s: hits train {sum((s >= 0).sum() for *_, s in sets['train'])} "
          f"sel {sum((s >= 0).sum() for *_, s in sets['sel'])} test {sum((s >= 0).sum() for *_, s in sets['test'])}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    tag = os.environ.get("TAG", "stroke3d")
    if os.environ.get("SWEEP") == "1":
        res = {}
        for name, c in CONFIGS.items():
            cfg = {**DEFAULT, **c}
            _, ep, log = train(cfg, sets["train"], n_cls, 0, dev, sets["sel"])
            b = log[ep]
            res[name] = {"cfg": cfg, "epoch": ep, "sel_acc": b["sel_acc"], "sel_f1": b["sel_f1"], "log": log}
            print(f"SWEEP {name}: best sel macroF1={b['sel_f1']:.3f} acc={b['sel_acc']:.3f} at epoch {ep}", flush=True)
        json.dump(res, open(os.path.join(OUT, tag + "_sweep.json"), "w"), indent=1)
        return
    c = os.environ.get("CFG", "w24_d192_l4")
    cfg = {**DEFAULT, **(CONFIGS[c] if c in CONFIGS else json.loads(c))}
    refit = os.environ.get("REFIT") == "1"
    Xt, Mt, Yt = windows(sets["test"], cfg["W"])
    runs = []
    for seed in [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]:
        model, ep, log = train(cfg, sets["train"], n_cls, seed, dev, sets["sel"])
        if refit:
            model, _, _ = train(cfg, sets["train"] + sets["sel"], n_cls, seed, dev, None, epochs=ep + 1)
        acc, f1 = metrics(predict(model, Xt, Mt, dev), Yt)
        runs.append({"seed": seed, "epoch": ep, "test_acc": acc, "test_macro_f1": f1, "sel": log[ep]})
        print(f"RESULT {tag} seed {seed}: epoch {ep} sel macroF1={log[ep]['sel_f1']:.3f} | test at true hits acc={acc:.3f} macroF1={f1:.3f} (n={len(Yt)})", flush=True)
    A, Fm = np.array([r["test_acc"] for r in runs]), np.array([r["test_macro_f1"] for r in runs])
    print(f"MEAN {tag}: acc {100 * A.mean():.1f} +- {100 * A.std(ddof=1) if len(A) > 1 else 0:.1f}, macroF1 {100 * Fm.mean():.1f} +- {100 * Fm.std(ddof=1) if len(A) > 1 else 0:.1f}", flush=True)
    json.dump({"tag": tag, "cfg": cfg, "refit": refit, "feat": BS.FEAT, "data": tf.BASE, "runs": runs},
              open(os.path.join(OUT, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
