"""Dedicated hit detector over the 3D reconstruction, on the eval of tools/bfmd_stroke.py (PROTOCOL=clean): same KEEP
rallies, v1 rally filter, test / selection / training matches, per-frame features (v1's make_features of the chosen
representation, FEAT points appended as bfmd_stroke.py does) and v1's hit metric unchanged: peaks of the per-frame hit
probability (scipy find_peaks, height = threshold, distance 10), matched one-to-one to labelled hits within +-10 frames,
mirror-ensembled at test as v1's predict_probs.

Unlike v1's model (whole rally, hit + stroke heads) this is hit-only: frame tokens (linear + LayerNorm) -> residual
temporal convolutions -> pre-norm transformer encoder over a crop of C frames -> per-frame hit logit. Training: random
crops, left-right mirroring, BCE on a hit target (TGT=v1: v1's +-10 triangle; TGT=g<s>: Gaussian of s frames), positive
weight as v1, AdamW + OneCycle. Inference: overlapping crops, probabilities averaged. Epoch and peak threshold are picked
on the selection matches (hit F1); the test matches are scored once. REFIT=1 retrains on training + selection matches.

STUDENT_HIT=1 scores the student's own hit head instead (no training; SD from rallynet_bfmd_export.py OWN=1);
HITP=1 adds that hit probability as one more input channel.
env as bfmd_stroke.py for the data (BFMD_DATA_ROOT, FEAT, SD, KEEP); CFG json overrides DEFAULT; SEEDS; TAG
out outputs/bfmd/hit3d/<TAG>.json"""
import copy, json, os, sys, time

os.environ.setdefault("PROTOCOL", "clean")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import stroke3d as S3  # noqa: E402  (load(): per side (features, mirrored features, stroke label per frame))

OUT = os.path.join(ROOT, "outputs", "bfmd", "hit3d")
DEFAULT = dict(C=256, d=192, L=4, heads=4, conv=2, drop=0.1, epochs=80, bs=32, lr=5e-4, wd=0.05, tgt="v1",
               crops=2)
TOL = 10
THRESHOLDS = [round(t, 2) for t in np.arange(0.1, 0.91, 0.05)]


def target(shot, kind):
    hits = np.flatnonzero(shot >= 0)
    t = np.arange(len(shot))[:, None]
    if not len(hits):
        return np.zeros(len(shot), np.float32)
    d = np.abs(t - hits[None]).min(1).astype(np.float32)
    if kind == "v1":
        return np.clip(1 - 0.1 * d, 0, 1) * (d <= 8)  # v1: 1.0 at the hit, 0.1 less per frame, 0.2 at +-8
    s = float(kind[1:])
    return np.exp(-0.5 * (d / s) ** 2).astype(np.float32)


class ConvBlock(nn.Module):
    def __init__(self, d, drop):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 5, padding=2, groups=1)
        self.drop = nn.Dropout(drop)

    def forward(self, h):
        return h + self.drop(F.gelu(self.conv(self.norm(h).transpose(1, 2)).transpose(1, 2)))


class HitNet(nn.Module):
    def __init__(self, D, cfg):
        super().__init__()
        d = cfg["d"]
        self.inp = nn.Sequential(nn.Linear(D, d), nn.LayerNorm(d))
        self.convs = nn.Sequential(*[ConvBlock(d, cfg["drop"]) for _ in range(cfg["conv"])])
        self.pos = nn.Parameter(torch.zeros(1, cfg["C"], d))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], 4 * d, cfg["drop"], batch_first=True, norm_first=True,
                                           activation="gelu")
        self.enc = nn.TransformerEncoder(layer, cfg["L"])
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, x, m):
        h = self.convs(self.inp(x)) + self.pos[:, :x.shape[1]]
        h = self.enc(h, src_key_padding_mask=~m)
        return self.out(h)[..., 0]


def crops(items, C, n_per, rng, mirror_p=0.5, kind="v1"):
    X, M, Y = [], [], []
    for feat, mir, shot in items:
        T = len(feat)
        tgt = target(shot, kind)
        for _ in range(n_per):
            s = int(rng.integers(0, max(1, T - C + 1)))
            src = mir if rng.random() < mirror_p else feat
            x = np.zeros((C, feat.shape[1]), np.float32); y = np.zeros(C, np.float32); m = np.zeros(C, bool)
            n = min(C, T - s)
            x[:n], y[:n], m[:n] = src[s:s + n], tgt[s:s + n], True
            X.append(x); Y.append(y); M.append(m)
    return np.stack(X), np.stack(M), np.stack(Y)


def probs(model, feat, C, dev):
    """Per-frame hit probability over a whole sequence: crops of C frames at stride C/4, probabilities averaged."""
    T = len(feat)
    starts = list(range(0, max(1, T - C + 1), C // 4))
    if starts[-1] + C < T:
        starts.append(T - C)
    X = np.zeros((len(starts), C, feat.shape[1]), np.float32); M = np.zeros((len(starts), C), bool)
    for i, s in enumerate(starts):
        n = min(C, T - s)
        X[i, :n], M[i, :n] = feat[s:s + n], True
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(X).to(dev), torch.from_numpy(M).to(dev))).cpu().numpy()
    acc, cnt = np.zeros(T), np.zeros(T)
    for i, s in enumerate(starts):
        n = min(C, T - s)
        acc[s:s + n] += p[i, :n]; cnt[s:s + n] += 1
    return acc / cnt


def all_probs(model, items, C, dev, mirror=True):
    model.eval()
    return [(probs(model, f, C, dev) + probs(model, mr, C, dev)) / 2 if mirror else probs(model, f, C, dev)
            for f, mr, _ in items]


def score(P, items, thr):
    """v1's evaluate_peaks, hit part: (P, R, F1, tp, fp, fn)."""
    tp = fp = fn = 0
    for p, (_, _, shot) in zip(P, items):
        true = list(np.flatnonzero(shot >= 0))
        peaks, _ = find_peaks(p, height=thr, distance=TOL)
        matched = set()
        for q in peaks:
            cand = [f for f in true if abs(f - q) <= TOL and f not in matched]
            if cand:
                matched.add(min(cand, key=lambda f: abs(f - q))); tp += 1
            else:
                fp += 1
        fn += len(true) - len(matched)
    P_ = tp / (tp + fp) if tp + fp else 0.0
    R_ = tp / (tp + fn) if tp + fn else 0.0
    return P_, R_, (2 * P_ * R_ / (P_ + R_) if P_ + R_ else 0.0), tp, fp, fn


def best_thr(P, items):
    return max(((score(P, items, t)[2], t) for t in THRESHOLDS))


def train(cfg, items, seed, dev, sel=None, epochs=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    epochs = epochs or cfg["epochs"]
    model = HitNet(items[0][0].shape[1], cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    allt = np.concatenate([target(s, cfg["tgt"]) for *_, s in items])
    pos_frac = (allt > 0.5).mean()
    pos_w = min(10.0, (1 - pos_frac) / max(pos_frac, 1e-4))
    nb = len(items) * cfg["crops"] // cfg["bs"]
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=max(epochs * nb, 1), pct_start=0.1)
    best, best_ep, best_t, best_state, log = -1, -1, 0.5, None, []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        X, M, Y = crops(items, cfg["C"], cfg["crops"], rng, kind=cfg["tgt"])
        perm = rng.permutation(len(X))
        tot = 0
        for b in range(nb):
            i = perm[b * cfg["bs"]:(b + 1) * cfg["bs"]]
            x, m, y = (torch.from_numpy(a[i]).to(dev) for a in (X, M, Y))
            out = model(x, m)
            w = 1 + (pos_w - 1) * y
            loss = (F.binary_cross_entropy_with_logits(out, y, reduction="none") * w * m).sum() / m.sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        msg = f"epoch {ep:2d}: loss {tot / max(nb, 1):.4f} ({time.time() - t0:.0f}s)"
        if sel is not None and (ep % 2 == 1 or ep == epochs - 1):
            f1, t = best_thr(all_probs(model, sel, cfg["C"], dev), sel)
            log.append({"epoch": ep, "sel_f1": f1, "thr": t})
            msg += f" | sel F1={f1:.4f} @ {t}"
            if f1 > best:
                best, best_ep, best_t, best_state = f1, ep, t, copy.deepcopy(model.state_dict())
        print(msg, flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_ep, best_t, log


def side_keys():
    """(tag, side) of every item of S3.load(), in its order: sorted tags, v1's rally filter, top then bottom."""
    BS = S3.BS
    d = json.load(open(f"{S3.tf.BASE}/training_dataset_all_selfdetected.json"))
    keep = set(open(BS.KEEP).read().split())
    d["rallies"] = {t: r for t, r in d["rallies"].items() if t in keep}
    _, test_tags, test_m = S3.tf.load_split(d)
    mor = d["match_of_rally"]
    tr_m, sel_m = BS.split_matches(mor, set(mor.values()) - test_m)
    tags = sorted(d["rallies"])
    ok = lambda t: d["rallies"][t]["nan_frac_top"] <= 0.5 and d["rallies"][t]["nan_frac_bottom"] <= 0.5
    keys = {"train": [t for t in tags if mor[t] in tr_m], "sel": [t for t in tags if mor[t] in sel_m], "test": test_tags}
    return {k: [(t, s) for t in v if ok(t) for s in ("top", "bottom")] for k, v in keys.items()}


def student_hits(sets):
    """The student's own per-side hit probability (rallynet_bfmd_export.py OWN=1: hit_top / hit_bottom in SD)."""
    keys, cache, out = side_keys(), {}, {}
    for name, items in sets.items():
        assert len(keys[name]) == len(items), (name, len(keys[name]), len(items))
        P = []
        for (tag, side), (feat, _, _) in zip(keys[name], items):
            m, rt = tag.split("::")
            if m not in cache:
                cache[m] = json.load(open(os.path.join(S3.BS.SRC["sd"], m, "joints_selfdetected.json")))["rallies"]
            h = np.array(cache[m][rt].get("hit_" + side, []), np.float32)
            p = np.zeros(len(feat), np.float32)
            p[:min(len(p), len(h))] = h[:len(p)]
            P.append(p)
        out[name] = P
    return out


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    sets, _ = S3.load()
    n = lambda it: sum(int((s >= 0).sum()) for *_, s in it)
    print(f"hits train {n(sets['train'])} sel {n(sets['sel'])} test {n(sets['test'])}", flush=True)
    if os.environ.get("STUDENT_HIT") == "1" or os.environ.get("HITP") == "1":
        H = student_hits(sets)
        if os.environ.get("STUDENT_HIT") == "1":  # the student's hit head alone: threshold on selection, test once
            f1s, thr = best_thr(H["sel"], sets["sel"])
            p, r, f1, tp, fp, fn = score(H["test"], sets["test"], thr)
            print(f"RESULT student hit head: thr {thr} sel F1={f1s:.4f} | test P={p:.3f} R={r:.3f} F1={f1:.4f} "
                  f"(tp={tp} fp={fp} fn={fn})", flush=True)
            return
        # HITP=1: the student's hit probability as one more input channel (mirroring leaves it unchanged)
        sets = {k: [(np.concatenate([f, h[:, None]], 1), np.concatenate([mr, h[:, None]], 1), s)
                    for (f, mr, s), h in zip(v, H[k])] for k, v in sets.items()}
    tag = os.environ.get("TAG", "hit3d")
    if os.environ.get("SWEEP"):  # json list of CFG overrides: selection F1 only, seed 0, no test
        res = []
        for over in json.loads(os.environ["SWEEP"]):
            cfg = {**DEFAULT, **over}
            _, ep, thr, log = train(cfg, sets["train"], 0, dev, sets["sel"])
            res.append({"cfg": over, "sel_f1": max(l["sel_f1"] for l in log), "epoch": ep, "thr": thr})
            print(f"SWEEP {json.dumps(over)}: sel F1={res[-1]['sel_f1']:.4f} epoch {ep} thr {thr}", flush=True)
        os.makedirs(OUT, exist_ok=True)
        json.dump(res, open(os.path.join(OUT, tag + "_sweep.json"), "w"), indent=1)
        return
    cfg = {**DEFAULT, **json.loads(os.environ.get("CFG", "{}"))}
    refit = os.environ.get("REFIT") == "1"
    runs = []
    for seed in [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]:
        model, ep, thr, log = train(cfg, sets["train"], seed, dev, sets["sel"])
        sel_f1 = max(l["sel_f1"] for l in log)
        if refit:
            model, _, _, _ = train(cfg, sets["train"] + sets["sel"], seed, dev, None, epochs=ep + 1)
        p, r, f1, tp, fp, fn = score(all_probs(model, sets["test"], cfg["C"], dev), sets["test"], thr)
        runs.append({"seed": seed, "epoch": ep, "thr": thr, "sel_f1": sel_f1, "P": p, "R": r, "F1": f1,
                     "tp": tp, "fp": fp, "fn": fn, "log": log})
        print(f"RESULT {tag} seed {seed}: epoch {ep} thr {thr} sel F1={sel_f1:.4f} | test P={p:.3f} R={r:.3f} "
              f"F1={f1:.4f} (tp={tp} fp={fp} fn={fn})", flush=True)
    A = np.array([r["F1"] for r in runs])
    print(f"MEAN {tag}: hit F1 {100 * A.mean():.1f} +- {100 * A.std(ddof=1) if len(A) > 1 else 0:.1f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"tag": tag, "cfg": cfg, "refit": refit, "feat": os.environ.get("FEAT"), "sd": os.environ.get("SD"),
               "data": S3.tf.BASE, "runs": runs}, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
