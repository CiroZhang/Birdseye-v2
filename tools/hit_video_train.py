"""Video hit model: is there a racket hit at the centre frame of this 8-frame window, and on which side of the net?
Input: the synced 2x2 mosaic (side | player cam 1 / player cam 2 | TV, 256x144 each, tools/hit_clips.py), frames
c-4..c+3. Each view goes through an ImageNet ResNet-18 (shared), view features are concatenated per frame, then a
temporal conv stack; heads: hit logit for the centre frame, side logit (near / far, positives only).
Labels: physics-verified teacher hits (centre within 1 frame -> positive; 4-12 frames away -> hard negative) and
confident mid-flight frames (negative). The hand answer-key rallies are not in the training data.
Modes: train (writes ckpt), infer <mosaic dir> <out dir> (per-frame hit and side probabilities for every rally).
usage: python hit_video_train.py train <data dir> <out dir> | infer <ckpt> <mosaic dir> <out dir>"""
import json, os, random, sys, time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

T, HALF = 8, 4
VW, VH = 256, 144


def read_window(path, c, n):
    """frames c-4..c+3 (clamped) of the mosaic -> uint8 (T, 288, 512, 3) RGB"""
    cap = cv2.VideoCapture(path)
    s = max(0, min(c - HALF, n - T))
    cap.set(cv2.CAP_PROP_POS_FRAMES, s)
    out = []
    for _ in range(T):
        ok, im = cap.read()
        if not ok:
            im = out[-1] if out else np.zeros((2 * VH, 2 * VW, 3), np.uint8)
        else:
            im = im[:, :, ::-1]
        out.append(im)
    a = np.stack(out)
    idx = np.clip(np.arange(c - HALF, c - HALF + T), 0, n - 1) - s   # edge frames repeat, as infer() pads
    return np.ascontiguousarray(a[idx])


class Data(torch.utils.data.Dataset):
    def __init__(s, root, uids, lab, n_neg=40, seed=0):
        rng = random.Random(seed); s.root = root; s.items = []
        for u in uids:
            d = lab[u]; n = d["n"]
            hits = [h for h, _ in d["pos"]]
            for h, side in d["pos"]:
                for o in (-1, 0, 1):
                    c = int(round(h)) + o
                    if 0 <= c < n:
                        s.items.append((u, n, c, 1.0, side))
                for o in rng.sample([-12, -10, -8, -6, -5, -4, 4, 5, 6, 8, 10, 12], 3):   # timing: near a hit but not on it
                    c = int(round(h)) + o
                    if 0 <= c < n and all(abs(c - g) > 3 for g in hits):
                        s.items.append((u, n, c, 0.0, -1))
            for c in rng.sample(d["neg"], min(n_neg, len(d["neg"]))):
                if 0 <= c < n:
                    s.items.append((u, n, c, 0.0, -1))

    def __len__(s):
        return len(s.items)

    def __getitem__(s, i):
        u, n, c, y, side = s.items[i]
        a = read_window(os.path.join(s.root, u + ".mp4"), c, n)
        if random.random() < 0.5:                                   # mild colour jitter
            a = np.clip(a.astype(np.float32) * random.uniform(0.8, 1.2) + random.uniform(-15, 15), 0, 255).astype(np.uint8)
        return torch.from_numpy(a), torch.tensor(y), torch.tensor(side)


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        s.enc = nn.Sequential(*list(r.children())[:-1])            # -> 512
        s.view_emb = nn.Parameter(torch.zeros(4, 512))
        s.tmp = nn.Sequential(nn.Conv1d(4 * 512, 512, 3, padding=1), nn.GELU(), nn.Conv1d(512, 256, 3, padding=1), nn.GELU())
        s.hit = nn.Linear(256 * T, 1); s.side = nn.Linear(256 * T, 1)
        s.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255)
        s.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255)

    def forward(s, x):                                             # x (B, T, 288, 512, 3) uint8
        B = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).float()                       # B T 3 H W
        views = [x[..., :VH, :VW], x[..., :VH, VW:], x[..., VH:, :VW], x[..., VH:, VW:]]
        v = torch.stack(views, 2).reshape(B * T * 4, 3, VH, VW)
        f = s.enc((v - s.mean) / s.std).flatten(1).view(B, T, 4, 512) + s.view_emb
        f = s.tmp(f.view(B, T, 4 * 512).transpose(1, 2)).flatten(1)
        return s.hit(f)[:, 0], s.side(f)[:, 0]


def train(root, out):
    lab = json.load(open(os.path.join(root, os.environ.get("LABELS", "labels.json"))))
    uids = sorted(u for u, d in lab.items() if "error" not in d and os.path.exists(os.path.join(root, u + ".mp4")))
    random.Random(1).shuffle(uids); nv = max(20, len(uids) // 20)
    va, tr = uids[:nv], uids[nv:]
    seed = int(os.environ.get("SEED", "0")); torch.manual_seed(seed); random.seed(seed)   # ensemble members differ by seed
    dtr, dva = Data(root, tr, lab, seed=seed), Data(root, va, lab, n_neg=15, seed=1)
    print(f"train {len(tr)} rallies {len(dtr)} windows ({sum(i[3] for i in dtr.items):.0f} pos); val {len(va)} rallies {len(dva)} windows", flush=True)
    L = lambda d, sh: torch.utils.data.DataLoader(d, batch_size=int(os.environ.get("BS", "32")), shuffle=sh, num_workers=int(os.environ.get("NW", "14")), drop_last=sh, persistent_workers=True)
    ltr, lva = L(dtr, True), L(dva, False)
    dev = "cuda"; net = Net().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([{"params": net.enc.parameters(), "lr": 1e-4}, {"params": [p for n_, p in net.named_parameters() if not n_.startswith("enc.")], "lr": 5e-4}], weight_decay=1e-4)
    E = int(os.environ.get("EPOCHS", "6")); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[1e-4, 5e-4], total_steps=E * len(ltr))
    bce = nn.BCEWithLogitsLoss(reduction="none"); best = 1e9; os.makedirs(out, exist_ok=True)
    for ep in range(E):
        net.train(); t0 = time.time(); tl = 0
        for i, (x, y, side) in enumerate(ltr):
            x, y, side = x.to(dev, non_blocking=True), y.to(dev).float(), side.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h, sd = net(x)
            loss = bce(h.float(), y).mean()
            m = side >= 0
            if m.any():
                loss = loss + 0.5 * bce(sd.float()[m], side[m].float()).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); tl += float(loss)
            if i % 200 == 0:
                print(f"ep {ep} it {i}/{len(ltr)} loss {tl / (i + 1):.4f} {time.time() - t0:.0f}s", flush=True)
        net.eval(); P, Y, S, SP = [], [], [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y, side in lva:
                h, sd = net(x.to(dev)); P.append(torch.sigmoid(h.float()).cpu()); Y.append(y); S.append(side); SP.append(torch.sigmoid(sd.float()).cpu())
        P, Y, S, SP = torch.cat(P), torch.cat(Y), torch.cat(S), torch.cat(SP)
        vl = float(nn.functional.binary_cross_entropy(P.clamp(1e-4, 1 - 1e-4), Y.float()))
        acc = float(((P > 0.5).float() == Y).float().mean()); rec = float((P[Y == 1] > 0.5).float().mean()); fpr = float((P[Y == 0] > 0.5).float().mean())
        sacc = float(((SP[S >= 0] > 0.5).long() == S[S >= 0]).float().mean())
        print(f"== ep {ep}: val loss {vl:.4f} acc {acc:.3f} hit recall {rec:.3f} false-pos rate {fpr:.4f} side acc {sacc:.3f}", flush=True)
        torch.save(net.state_dict(), os.path.join(out, "last.pt"))
        if vl < best:
            best = vl; torch.save(net.state_dict(), os.path.join(out, "best.pt"))


def infer(ckpt, root, out):
    """per-frame hit and side probability for every mosaic in root (sliding window, every frame)"""
    dev = "cuda"; net = Net().to(dev); net.load_state_dict(torch.load(ckpt, map_location=dev)); net.eval()
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(root)):
        if not f.endswith(".mp4"):
            continue
        cap = cv2.VideoCapture(os.path.join(root, f)); fr = []
        while True:
            ok, im = cap.read()
            if not ok:
                break
            fr.append(im[:, :, ::-1])
        a = np.stack(fr); n = len(a); pad = np.concatenate([a[:1].repeat(HALF, 0), a, a[-1:].repeat(T - HALF, 0)])
        ph, ps = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for b in range(0, n, 64):
                x = torch.from_numpy(np.stack([pad[c:c + T] for c in range(b, min(n, b + 64))])).to(dev)
                h, sd = net(x); ph.append(torch.sigmoid(h.float()).cpu().numpy()); ps.append(torch.sigmoid(sd.float()).cpu().numpy())
        np.savez(os.path.join(out, f[:-4] + ".npz"), hit=np.concatenate(ph), far=np.concatenate(ps))
        print(f, n, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3])
    else:
        infer(sys.argv[2], sys.argv[3], sys.argv[4])
