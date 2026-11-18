"""TV-only body placement + racket student, trained on the four-camera teacher (arrays from tools/tvpose_data.py).

A temporal transformer over each player's track: inputs are TV VIMO joints (pelvis-relative), the pelvis, the TV view
and the TV racket box near the hands; outputs per frame are the pelvis correction (to the teacher's multi-camera
pelvis), the racket grip offset from the hand, the racket's unit direction, and a racket-hand logit.

usage: python tools/tvpose_train.py train DATA.npz OUT.pt [--hold uids.txt] [--epochs N] [--d 256] [--layers 4]
       python tools/tvpose_train.py infer MODEL.pt DATA.npz OUT.npz
"""
import argparse, hashlib, math, os, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WIN = 96
MOT = float(os.environ.get("TP_MOT", "2.0"))  # weight of the placement motion loss (0 = off)


def is_val(uid):
    return int(hashlib.md5(uid.encode()).hexdigest(), 16) % 10 == 0


class Net(nn.Module):
    def __init__(self, nf, d=256, layers=4, heads=8, drop=0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(nf, d), nn.GELU(), nn.Linear(d, d))
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(el, layers)
        self.out = nn.Linear(d, 3 + 3 + 3 + 1)
        pe = torch.zeros(4096, d)
        p = torch.arange(4096).float()[:, None]
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(p * div), torch.cos(p * div)
        self.register_buffer("pe", pe)

    def forward(self, x, pad):
        h = self.inp(x) + self.pe[: x.shape[1]][None]
        h = self.enc(h, src_key_padding_mask=pad)
        o = self.out(h)
        return o[..., :3], o[..., 3:6], F.normalize(o[..., 6:9], dim=-1), o[..., 9]


def run_windows(net, X, pad):
    """the network over a whole track in WIN-frame windows (half overlap, tapered blend): it is trained on WIN frames."""
    T = X.shape[1]
    if T <= WIN:
        return net(X, pad)
    acc, wsum = None, torch.zeros(T, device=X.device)
    taper = torch.minimum(torch.arange(1, WIN + 1, device=X.device), torch.arange(WIN, 0, -1, device=X.device)).float()
    starts = list(range(0, T - WIN + 1, int(os.environ.get("TP_STRIDE", WIN // 2))))  # denser overlap = more windows averaged
    if starts[-1] != T - WIN:
        starts.append(T - WIN)
    for s in starts:
        o = torch.cat([v if v.dim() == 3 else v[..., None] for v in net(X[:, s:s + WIN], pad[:, s:s + WIN])], -1)[0]
        if acc is None:
            acc = torch.zeros(T, o.shape[-1], device=X.device)
        acc[s:s + WIN] += o * taper[:, None]; wsum[s:s + WIN] += taper
    o = (acc / wsum[:, None])[None]
    return o[..., :3], o[..., 3:6], F.normalize(o[..., 6:9], dim=-1), o[..., 9]


def load(path):
    z = dict(np.load(path))  # decompress every array once (indexing the npz per track re-reads it each time)
    off = z["off"]
    tr = []
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        tr.append(dict(uid=str(z["uid"][i]), slot=str(z["slot"][i]), hand=float(z["hand"][i]),
                       **{k: z[k][a:b] for k in ("X", "ok", "frames", "J", "dp", "dp_m", "grip", "u", "rk_w")}))
    return tr


def stats(tracks):
    X = np.concatenate([t["X"][t["ok"]].astype(np.float32) for t in tracks])
    mu, sd = X.mean(0), X.std(0) + 1e-3
    mu[78:], sd[78:] = 0, 1  # camera axes and box features stay as they are
    return mu, sd


def batch(tracks, idx, mu, sd, dev, train):
    xs, rows = [], []
    for i in idx:
        t = tracks[i]
        T = len(t["X"])
        s = np.random.randint(0, max(1, T - WIN + 1)) if train else 0
        e = min(T, s + WIN) if train else T
        rows.append((t, s, e))
    L = max(e - s for _, s, e in rows)
    B = len(rows)
    X = np.zeros((B, L, tracks[0]["X"].shape[1]), np.float32)
    pad = np.ones((B, L), bool)
    tg = {k: np.zeros((B, L, 3), np.float32) for k in ("dp", "grip", "u", "p0")}
    mk = {k: np.zeros((B, L), np.float32) for k in ("dp_m", "rk_w", "ok")}
    hand = np.zeros((B,), np.float32)
    for j, (t, s, e) in enumerate(rows):
        n = e - s
        x = (t["X"][s:e].astype(np.float32) - mu) / sd
        x[~t["ok"][s:e]] = 0
        if train and np.random.rand() < 0.3:   # sometimes no racket boxes at all (the TV often misses them)
            x[:, 84:91] = 0
        X[j, :n] = x
        pad[j, :n] = False
        for k in ("dp", "grip", "u"):
            tg[k][j, :n] = t[k][s:e]
        tg["p0"][j, :n] = np.nan_to_num(t["J"][s:e, 0])  # VIMO pelvis: placement = p0 + dp
        mk["dp_m"][j, :n] = t["dp_m"][s:e]; mk["rk_w"][j, :n] = t["rk_w"][s:e]; mk["ok"][j, :n] = t["ok"][s:e]
        hand[j] = t["hand"]
    T_ = lambda a: torch.from_numpy(a).to(dev)
    return T_(X), T_(pad), {k: T_(v) for k, v in tg.items()}, {k: T_(v) for k, v in mk.items()}, T_(hand)


def losses(net, X, pad, tg, mk, hand):
    dp, grip, u, hl = net(X, pad)
    m = mk["dp_m"] * mk["ok"]
    l_dp = (F.huber_loss(dp, tg["dp"], delta=0.2, reduction="none").sum(-1) * m).sum() / (m.sum() + 1)
    w = mk["rk_w"] * mk["ok"]
    l_g = (F.huber_loss(grip, tg["grip"], delta=0.05, reduction="none").sum(-1) * w).sum() / (w.sum() + 1)
    l_u = ((1 - (u * tg["u"]).sum(-1)) * w).sum() / (w.sum() + 1)
    # motion: the placed pelvis (VIMO pelvis + correction) moves like the teacher's from frame to frame (velocity and
    # acceleration), not just near it on average -- a single TV view gives noisy depth that otherwise shows up as shake
    P, Pt = tg["p0"] + dp, tg["p0"] + tg["dp"]
    m1 = m[:, 1:] * m[:, :-1]; m2 = m1[:, 1:] * m1[:, :-1]
    d1 = lambda A: A[:, 1:] - A[:, :-1]
    nrm = lambda A: torch.sqrt((A * A).sum(-1) + 1e-8)
    l_v = (nrm(d1(P) - d1(Pt)) * m1).sum() / (m1.sum() + 1)
    l_a = (nrm(d1(d1(P)) - d1(d1(Pt))) * m2).sum() / (m2.sum() + 1)
    l_mot = MOT * (l_v + l_a)
    hm = (hand >= 0).float()[:, None] * mk["ok"]
    l_h = (F.binary_cross_entropy_with_logits(hl, hand.clamp(min=0)[:, None].expand_as(hl), reduction="none") * hm).sum() / (hm.sum() + 1)
    return l_dp + l_mot + l_g + 2 * l_u + 0.1 * l_h, dict(dp=l_dp.item(), mot=l_mot.item(), g=l_g.item(), u=l_u.item(), h=l_h.item())


def forearm(t):
    """the VIMO forearm direction of the racket hand (the teacher's prior where no camera sees the racket)."""
    J = t["J"]; hi = int(max(t["hand"], 0))
    f = J[:, 20 + hi] - J[:, 18 + hi]
    return f / (np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9)


@torch.no_grad()
def evaluate(net, tracks, mu, sd, dev):
    net.eval()
    e_dp0, e_dp, a_u0, a_u, a_u_obs, a_u0_obs, hand_ok, j_net, j_tch = [], [], [], [], [], [], [], [], []
    for i in range(len(tracks)):
        t = tracks[i]
        X, pad, tg, mk, hand = batch(tracks, [i], mu, sd, dev, False)
        dp, grip, u, hl = run_windows(net, X, pad)
        dp, u = dp[0].cpu().numpy(), u[0].cpu().numpy()
        m = (t["dp_m"] > 0) & t["ok"]
        e_dp0 += list(np.linalg.norm(t["dp"][m], axis=1)); e_dp += list(np.linalg.norm(t["dp"][m] - dp[m], axis=1))
        m3 = m[2:] & m[1:-1] & m[:-2]
        acc = lambda Q: np.linalg.norm(Q[2:] - 2 * Q[1:-1] + Q[:-2], axis=-1)[m3]
        p0 = np.nan_to_num(t["J"][:, 0])
        j_net += list(acc(p0 + dp)); j_tch += list(acc(p0 + t["dp"]))
        w = (t["rk_w"] > 0) & t["ok"]; wo = (t["rk_w"] == 1) & t["ok"]
        ang = lambda a, b: np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
        fa = forearm(t)
        a_u += list(ang(u[w], t["u"][w])); a_u0 += list(ang(fa[w], t["u"][w]))
        a_u_obs += list(ang(u[wo], t["u"][wo])); a_u0_obs += list(ang(fa[wo], t["u"][wo]))
        if t["hand"] >= 0:
            hand_ok.append(float((hl[0][t["ok"]].mean() > 0).item() == (t["hand"] > 0.5)))
    net.train()
    md = lambda a: float(np.median(a)) if len(a) else float("nan")
    return dict(pelvis_vimo=md(e_dp0), pelvis_net=md(e_dp), rk_forearm=md(a_u0), rk_net=md(a_u),
                rk_forearm_seen=md(a_u0_obs), rk_net_seen=md(a_u_obs), jit_net_cm=100 * md(j_net), jit_teacher_cm=100 * md(j_tch), hand_acc=float(np.mean(hand_ok)) if hand_ok else float("nan"))


def train(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tracks = load(a.data)
    hold = set(open(a.hold).read().split()) if a.hold else set()
    tr = [t for t in tracks if t["uid"] not in hold and not is_val(t["uid"])]
    va = [t for t in tracks if t["uid"] not in hold and is_val(t["uid"])]
    print(f"train {len(tr)} tracks, val {len(va)}, held out {sum(t['uid'] in hold for t in tracks)}", flush=True)
    mu, sd = stats(tr)
    net = Net(tr[0]["X"].shape[1], a.d, a.layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * max(1, len(tr) // a.bs)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    print("before", evaluate(net, va, mu, sd, dev), flush=True)
    st, t0, best = 0, time.time(), 1e9
    for ep in range(a.epochs):
        perm = np.random.permutation(len(tr))
        agg = {}
        for k in range(0, len(perm) - a.bs + 1, a.bs):
            X, pad, tg, mk, hand = batch(tr, perm[k:k + a.bs], mu, sd, dev, True)
            loss, parts = losses(net, X, pad, tg, mk, hand)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step(); st += 1
            for q, v in parts.items():
                agg[q] = agg.get(q, 0) + v
        n = max(1, len(perm) // a.bs)
        if ep % a.every == 0 or ep == a.epochs - 1:
            ev = evaluate(net, va, mu, sd, dev)
            score = ev["pelvis_net"] + ev["rk_net"] / 100
            if score < best:
                best = score
                torch.save(dict(sd=net.state_dict(), mu=mu, sdv=sd, d=a.d, layers=a.layers, nf=tr[0]["X"].shape[1]), a.out)
            print(f"ep {ep} {time.time() - t0:.0f}s " + " ".join(f"{q} {v / n:.4f}" for q, v in agg.items()) + " | " +
                  " ".join(f"{q} {v:.3f}" for q, v in ev.items()), flush=True)
    print("DONE best", best)


def infer(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    net = Net(ck["nf"], ck["d"], ck["layers"]).to(dev); net.load_state_dict(ck["sd"]); net.eval()
    tracks = load(a.data)
    if a.eval:
        print("eval", evaluate(net, tracks, ck["mu"], ck["sdv"], dev))
    out = dict(uid=[], slot=[], off=[0], frames=[], pelvis=[], grip=[], u=[], hand=[])
    with torch.no_grad():
        for i, t in enumerate(tracks):
            X, pad, *_ = batch(tracks, [i], ck["mu"], ck["sdv"], dev, False)
            dp, grip, u, hl = [v[0].cpu().numpy() for v in run_windows(net, X, pad)]
            hi = int(hl[t["ok"]].mean() > 0) if t["ok"].any() else 1
            fa = t["J"][:, 20 + hi] - t["J"][:, 18 + hi]
            fa /= np.linalg.norm(fa, axis=-1, keepdims=True) + 1e-9
            hand_pt = t["J"][:, 20 + hi] + 0.07 * fa + dp  # the body moves with its pelvis correction, the racket with it
            out["uid"].append(t["uid"]); out["slot"].append(t["slot"]); out["off"].append(out["off"][-1] + len(dp))
            out["frames"].append(t["frames"]); out["pelvis"].append(t["J"][:, 0] + dp)
            out["grip"].append(hand_pt + grip); out["u"].append(u); out["hand"].append(20 + hi)
    np.savez_compressed(a.out, uid=np.array(out["uid"]), slot=np.array(out["slot"]), off=np.array(out["off"]),
                        hand=np.array(out["hand"]), **{k: np.concatenate(out[k]) for k in ("frames", "pelvis", "grip", "u")})
    print("saved", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode")
    ap.add_argument("a1"); ap.add_argument("a2"); ap.add_argument("a3", nargs="?")
    ap.add_argument("--hold"); ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.mode == "train":
        a.data, a.out = a.a1, a.a2
        train(a)
    else:
        a.model, a.data, a.out = a.a1, a.a2, a.a3
        infer(a)
