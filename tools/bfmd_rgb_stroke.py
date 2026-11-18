"""RGB video baseline for BFMD stroke type, on exactly the eval of tools/bfmd_stroke.py (PROTOCOL=clean, SELECT=gtf1):
same KEEP rallies and v1 rally filter, same test matches (v1's load_split), same selection matches (split_matches on
the rest), and the same metrics at the labelled hit frames (accuracy, macro-F1 over the classes present).

Model: VideoMAE-Base, Kinetics-400 fine-tuned (MCG-NJU/videomae-base-finetuned-kinetics), fully fine-tuned. Input per
labelled hit: 16 full broadcast frames at stride 2 centred on the hit (frames f-16 .. f+14), resized to 224x224, plus
which player hits (top/bottom) as a learned embedding added to the pooled feature. Training: horizontal flip,
+-2 frame jitter of the centre. The epoch is picked on selection macro-F1; the test matches are scored once.

out  outputs/bfmd/stroke/<TAG>.json, checkpoint outputs/bfmd/stroke/<TAG>.pt
usage: python tools/bfmd_rgb_stroke.py   (env SEED=0, EPOCHS=12, BS=8, LR=5e-5, KEEP=outputs/stroke_keep_bfmd.txt,
       REFIT=1 retrains on train+selection matches for the picked number of epochs)"""
import json, os, sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "third_party", "birdseye_v1", "action"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_final as tf  # noqa: E402  (load_split only)

SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "12"))
BS = int(os.environ.get("BS", "8"))
LR = float(os.environ.get("LR", "5e-5"))
KEEP = os.environ.get("KEEP", os.path.join(ROOT, "outputs", "stroke_keep_bfmd.txt"))
REFIT = os.environ.get("REFIT") == "1"
SET = os.environ.get("SET", "bfmd")  # fbs: FineBadminton-20K (tools/fbs_export.py layout, clips data/fb20k/clips)
TAG = os.environ.get("TAG", f"{SET}_rgb_videomae{'_refit' if REFIT else ''}_s{SEED}")
DATA = os.path.join(ROOT, "data", *(("bfmd", "v2") if SET == "bfmd" else ("fbs", "2d")), "training_dataset_all_selfdetected.json")
CLIPS = os.path.join(ROOT, "data", *(("bfmd", "clips") if SET == "bfmd" else ("fb20k", "clips")))
OUT = os.path.join(ROOT, "outputs", "bfmd" if SET == "bfmd" else "fbs", "stroke")


def clip_name(tag):
    """BFMD match::gN_rNNN -> match__gN_rNNN; FineBadminton match::g1_rKKK -> match_KKK (tools/fbs_export.py)."""
    if SET == "bfmd":
        return tag.replace("::", "__")
    m, rt = tag.split("::")
    return f"{m}_{int(rt.split('_r')[1]):03d}"
MODEL = "MCG-NJU/videomae-base-finetuned-kinetics"
T, STRIDE, SIZE = 16, 2, 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def split_matches(matches):
    """tools/bfmd_stroke.split_matches (v1's rule on the non-test matches): (train, selection)."""
    matches = sorted(matches)
    perm = np.random.default_rng(0).permutation(len(matches))
    n_val = max(2, len(matches) // 5)
    return {matches[i] for i in perm[n_val:]}, {matches[i] for i in perm[:n_val]}


def hits(dataset, tags, shot_to_idx):
    """(clip path, frame, side 0=top/1=bottom, class, n_frames) for every labelled hit, with v1's rally filter."""
    out = []
    for tag in tags:
        r = dataset["rallies"][tag]
        if r["nan_frac_top"] > 0.5 or r["nan_frac_bottom"] > 0.5:
            continue
        path = os.path.join(CLIPS, clip_name(tag) + ".mp4")
        for side, key in ((0, "shot_type_top"), (1, "shot_type_bottom")):
            for f, s in enumerate(r[key]):
                c = shot_to_idx.get(s, -1)
                if c >= 0:
                    out.append((path, f, side, c, r["n_frames"]))
    return out


class HitClips(Dataset):
    def __init__(self, items, train):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, f, side, c, n = self.items[i]
        if self.train:
            f += np.random.randint(-2, 3)
        idx = np.clip(np.arange(f - T, f + T, STRIDE), 0, n - 1)
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx[0]))
        frames, k, last = {}, int(idx[0]), None
        want = set(idx.tolist())
        while k <= idx[-1]:
            ok, im = cap.read()
            if not ok:
                break
            if k in want:
                last = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (SIZE, SIZE), interpolation=cv2.INTER_AREA)
                frames[k] = last
            k += 1
        cap.release()
        clip = []
        for j in idx:  # frames past a short decode repeat the last one decoded
            last = frames.get(int(j), last)
            clip.append(last if last is not None else np.zeros((SIZE, SIZE, 3), np.uint8))
        x = (np.stack(clip).astype(np.float32) - MEAN) / STD
        if self.train and np.random.rand() < 0.5:
            x = x[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2))), side, c


class Net(nn.Module):
    def __init__(self, n_cls):
        super().__init__()
        from transformers import VideoMAEModel
        self.enc = VideoMAEModel.from_pretrained(MODEL)
        d = self.enc.config.hidden_size
        self.side = nn.Embedding(2, d)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_cls)

    def forward(self, x, side):
        h = self.enc(pixel_values=x).last_hidden_state.mean(1)
        return self.head(self.norm(h + self.side(side)))


def scores(model, loader, dev):
    model.eval()
    P, Y = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, s, y in loader:
            P.append(model(x.to(dev), s.to(dev)).argmax(1).cpu().numpy()); Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    f1 = []
    for c in np.unique(Y):
        tp, fp, fn = ((P == c) & (Y == c)).sum(), ((P == c) & (Y != c)).sum(), ((P != c) & (Y == c)).sum()
        f1.append(2 * tp / (2 * tp + fp + fn))
    return float((P == Y).mean()), float(np.mean(f1)), int(len(Y))


def seed_worker(w):
    np.random.seed((torch.initial_seed() + w) % (1 << 32))


def train(items, n_cls, epochs, dev, sel_loader=None, ckpt=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = Net(n_cls).to(dev)
    loader = DataLoader(HitClips(items, True), batch_size=BS, shuffle=True, num_workers=6, drop_last=True,
                        persistent_workers=True, worker_init_fn=seed_worker)
    params = [{"params": model.enc.parameters(), "lr": LR}, {"params": list(model.side.parameters()) + list(model.norm.parameters()) + list(model.head.parameters()), "lr": LR * 10}]
    opt = torch.optim.AdamW(params, weight_decay=0.05)
    steps = epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR, LR * 10], total_steps=steps, pct_start=0.1)
    best, best_ep, log = -1, -1, []
    for ep in range(epochs):
        model.train()
        tot = 0
        for i, (x, s, y) in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x.to(dev), s.to(dev)), y.to(dev), label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        msg = f"epoch {ep:2d}: train loss {tot / len(loader):.3f}"
        if sel_loader is not None:
            acc, f1, _ = scores(model, sel_loader, dev)
            log.append({"epoch": ep, "sel_acc": acc, "sel_f1": f1})
            msg += f" | sel acc={acc:.3f} macroF1={f1:.3f}"
            if f1 > best:
                best, best_ep = f1, ep
                torch.save(model.state_dict(), ckpt)
        print(msg, flush=True)
    return model, best_ep, log


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    dataset = json.load(open(DATA))
    keep = set(open(KEEP).read().split())
    dataset["rallies"] = {t: r for t, r in dataset["rallies"].items() if t in keep}
    for r in dataset["rallies"].values():  # only the labels are needed
        for k in [k for k in r if k.endswith(("_rel", "_vel"))]:
            del r[k]
    vocab = dataset["shot_type_vocab"]
    shot_to_idx = {s: i for i, s in enumerate(vocab)}
    _, test_tags, test_matches = tf.load_split(dataset)
    mor = dataset["match_of_rally"]
    tr_m, sel_m = split_matches(set(mor.values()) - test_matches)
    tags = sorted(dataset["rallies"])
    tr = hits(dataset, [t for t in tags if mor[t] in tr_m], shot_to_idx)
    if os.environ.get("LIMIT"):
        tr = tr[:int(os.environ["LIMIT"])]
    sel = hits(dataset, [t for t in tags if mor[t] in sel_m], shot_to_idx)
    te = hits(dataset, test_tags, shot_to_idx)
    print(f"{TAG}: train {len(tr)} sel {len(sel)} test {len(te)} hits; test {sorted(test_matches)}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    ckpt = os.path.join(OUT, TAG + ".pt")
    mk = lambda it: DataLoader(HitClips(it, False), batch_size=BS * 2, num_workers=6)
    model, best_ep, log = train(tr, len(vocab), EPOCHS, dev, mk(sel), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    if REFIT:
        model, _, _ = train(tr + sel, len(vocab), best_ep + 1, dev)
        torch.save(model.state_dict(), ckpt)
    acc, f1, n = scores(model, mk(te), dev)
    res = {"tag": TAG, "model": MODEL, "gt_acc": acc, "gt_macro_f1": f1, "gt_hits": n, "epoch": best_ep, "refit": REFIT,
           "seed": SEED, "test_matches": sorted(test_matches), "log": log}
    json.dump(res, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1)
    print(f"RESULT {TAG}: at true hits: acc={acc:.3f} macroF1={f1:.3f} (n={n}) epoch {best_ep}", flush=True)


if __name__ == "__main__":
    main()
