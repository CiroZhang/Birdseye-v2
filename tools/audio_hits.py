"""Physics teaches the ears: a racket-hit detector on the broadcast audio, trained only on the teacher's verified hits.

The arena sound has a sharp pop at every racket hit, but also shoe squeaks, shouts, the umpire and the crowd.
Labels come from the 3D teacher (outputs/cctv/<AH_TEACH or teach>): a hit is *verified* when the confident flight
before it and the confident flight after it meet there (teacher sd < SD on >= FRAC of frames 2-8 before and 2-8
after); its time is where the two fitted flights meet (sub-frame). Negatives are times inside confident flight,
farther than NEG_GAP from any teacher hit. No human labels.

Model: log-mel spectrogram (64 bands, 5 ms hop) -> small 1-D conv net -> hit logit every 5 ms. A positive is a
window of +-POS_W around the teacher's hit time (multiple-instance: its best frame should fire); negatives are
every frame of the negative stretches. Train on the split's train matches, test on its test matches.

Outputs outputs/cctv/audio_hits/{model.pt, report.json, probs/<uid>.npy (200 Hz hit probability)}.
usage: python tools/audio_hits.py train | eval | probs"""
import glob, json, os, sys, time

import numpy as np
import torch
import torch.nn as nn
import torchaudio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
AUD = os.path.join(ROOT, "outputs", "cctv", "audio")
TEACH = os.path.join(ROOT, "outputs", "cctv", os.environ.get("AH_TEACH", "teach"))
OUT = os.path.join(ROOT, "outputs", "cctv", "audio_hits", os.environ.get("AH_MODE", "av") + ("_perc" if os.environ.get("AH_PERC") == "1" else ""))
SR, HOP, FPS = 16000, 80, 25.0          # 5 ms frames (200 Hz)
AF = SR // HOP                          # audio frames per second
SD, FRAC = 0.06, 0.6
POS_W, NEG_GAP = 0.03, 0.15             # s
CTX = 64                                # audio frames each side fed to the net (0.32 s)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODE = os.environ.get("AH_MODE", "av")    # audio | vision | av (audio-visual)
NV = 12                                   # visual features per frame
torch.backends.cudnn.allow_tf32 = False

mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=512, win_length=400, hop_length=HOP, n_mels=64, f_min=60, f_max=7800)


PERC = os.environ.get("AH_PERC") == "1"   # percussive part only (harmonic-percussive separation drops squeaks, shouts, hum)


def feats(uid):
    a = torch.from_numpy(np.load(os.path.join(AUD, uid + ".npy")).astype(np.float32) / 32768.0)
    if PERC:
        import librosa
        S = librosa.stft(a.numpy(), n_fft=512, hop_length=HOP, win_length=400)
        _, P = librosa.decompose.hpss(S, kernel_size=(31, 31), margin=2.0)
        m = torch.log(torch.from_numpy((mel.mel_scale.fb.T.numpy() @ (np.abs(P) ** 2)).astype(np.float32)) + 1e-6)
    else:
        m = torch.log(mel(a) + 1e-6)                 # (64, T)
    m = ((m - m.mean(1, keepdim=True)) / (m.std(1, keepdim=True) + 1e-4)).numpy().astype(np.float32)
    if MODE == "audio":
        return m
    v = vis(uid, m.shape[1])
    return v if MODE == "vision" else np.concatenate([m, v])


def _track(path, shift=0):
    """TV-clock {frame: (u, v)} of a refined 2D shuttle track."""
    if not os.path.exists(path):
        return {}
    r = json.load(open(path))
    return {int(f) - shift: np.array(p, float) for f, p in zip(r["frames"], r["px"]) if p is not None and np.isfinite(p).all()}


def vis(uid, T):
    """(NV, T) visual features at audio rate from 25 fps cues: per camera (TV, side) shuttle seen, 2-D velocity, turn
    (change of direction over +-2 frames); players: fastest wrist (TV rays), shuttle-to-nearest-wrist angle."""
    nF = int(np.ceil(T / AF * FPS)) + 2
    V = np.zeros((NV, nF), np.float32)
    tri = os.path.join(ROOT, "outputs", "cctv", "tri", uid + ".json")
    sh = json.load(open(tri)).get("shift", 0) if os.path.exists(tri) else 0
    for c, (path, s) in enumerate(((os.path.join(ROOT, "outputs", "cctv", "refined", uid + ".json"), 0),
                                   (os.path.join(ROOT, "outputs", "cctv_side", "refined", uid + ".json"), sh))):
        tr = _track(path, s)
        for f in range(nF):
            if f in tr:
                V[4 * c, f] = 1
                a, b = tr.get(f - 2), tr.get(f + 2)
                if a is not None and b is not None:
                    v1, v2 = (tr[f] - a) / 2, (b - tr[f]) / 2
                    V[4 * c + 1:4 * c + 3, f] = np.clip((v2 + v1) / 2 / 20, -3, 3)
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 0.5 and n2 > 0.5:
                        V[4 * c + 3, f] = 1 - v1 @ v2 / (n1 * n2)          # 0 straight .. 2 reversal
    ob = os.path.join(ROOT, "outputs", "rally_obs", "cctv", uid + ".npz")
    if os.path.exists(ob):
        o = np.load(ob); kd, kc, sd_, sv = o["kp_dir"], o["kp_conf"], o["sh_dir"], o["sh_vis"]
        n = min(len(kd), nF)
        W = kd[:n, :, [9, 10]]; wc = kc[:n, :, [9, 10]] > 0.3            # wrists (rays from the TV camera)
        sp = np.zeros((n, 4, 2)); sp[1:] = np.degrees(np.arccos(np.clip((W[1:] * W[:-1]).sum(-1), -1, 1)))
        sp[~wc] = 0; sp[1:][~wc[:-1]] = 0
        s_sorted = np.sort(sp.reshape(n, -1), 1)
        V[8, :n] = np.clip(s_sorted[:, -1] / 0.5, 0, 6); V[9, :n] = np.clip(s_sorted[:, -2] / 0.5, 0, 6)
        ang = np.degrees(np.arccos(np.clip((W * sd_[:n, None, None]).sum(-1), -1, 1)))
        ang[~wc] = 99
        V[10, :n] = np.clip(ang.reshape(n, -1).min(1) / 2, 0, 10) * (sv[:n] > 0)
        V[11, :n] = sv[:n] > 0
    idx = np.clip((np.arange(T) / AF * FPS).astype(int), 0, nF - 1)
    return np.nan_to_num(V[:, idx], nan=0.0, posinf=0.0, neginf=0.0)  # missing players / rays -> 0


def labels(uid, T):
    """(verified hit times s, negative mask over audio frames) from the teacher."""
    p = os.path.join(TEACH, uid + ".json")
    if not os.path.exists(p):
        return None
    o = json.load(open(p)); fr = np.array(o["frames"], float); sd = np.array(o["sd"], float); hits = np.array(o["hits"], float)
    conf = {int(f): s < SD for f, s in zip(fr, sd)}
    ok = lambda fs: np.mean([conf.get(int(f), False) for f in fs]) >= FRAC
    ver = [h for h in hits if ok(np.arange(np.floor(h) - 8, np.floor(h) - 1)) and ok(np.arange(np.ceil(h) + 2, np.ceil(h) + 9))]
    neg = np.zeros(T, bool)
    t = np.arange(T) / AF
    cf = np.array([f for f, c in conf.items() if c], float)
    if len(cf):
        inflight = np.isin(np.round(t * FPS).astype(int), cf.astype(int))
        far = np.ones(T, bool)
        for h in hits:
            far &= np.abs(t - h / FPS) > NEG_GAP
        neg = inflight & far
    return np.array(ver) / FPS, neg


class Net(nn.Module):
    def __init__(s, c=96):
        super().__init__()
        L = []
        cin = {"audio": 64, "vision": NV, "av": 64 + NV}[MODE]
        for d in (1, 2, 4, 8, 16, 1):
            L += [nn.Conv1d(cin, c, 5, padding=2 * d, dilation=d), nn.BatchNorm1d(c), nn.GELU()]
            cin = c
        s.body = nn.Sequential(*L); s.head = nn.Conv1d(c, 1, 1)

    def forward(s, x):                  # (B, 64, T) -> (B, T) logits
        return s.head(s.body(x))[:, 0]


def split():
    """(train, validation) rallies: the split's train rallies with audio and teacher output; every 5th rally held out
    (the split's test match has no teacher output by design, so it is not used here)."""
    sp = json.load(open(os.path.join(ROOT, "outputs", "cctv", "split.json")))
    have = {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(AUD, "*.npy"))}
    tr = sorted(u for u in sp["train"] if u in have and os.path.exists(os.path.join(TEACH, u + ".json")))
    return [u for i, u in enumerate(tr) if i % 5], [u for i, u in enumerate(tr) if i % 5 == 0]


def dataset(uids):
    D = []
    for u in uids:
        F = feats(u); lab = labels(u, F.shape[1])
        if lab is None:
            continue
        D.append((u, F, lab[0], lab[1]))
    return D


def batches(D, n=64, W=2 * CTX + 64):
    """Random crops of W frames: half centred near a verified hit, half anywhere in negative stretches."""
    rng = np.random.default_rng()
    pos = [(i, h) for i, d in enumerate(D) for h in d[2]]
    negs = [(i, np.flatnonzero(d[3])) for i, d in enumerate(D) if d[3].any()]
    while True:
        X, P, N = [], [], []
        for k in range(n):
            if k % 2 == 0 and pos:
                i, h = pos[rng.integers(len(pos))]; c = int(h * AF) + rng.integers(-40, 41)
            else:
                i, nz = negs[rng.integers(len(negs))]; c = int(nz[rng.integers(len(nz))])
            F, hits, neg = D[i][1], D[i][2], D[i][3]
            s0 = c - W // 2; idx = np.clip(np.arange(s0, s0 + W), 0, F.shape[1] - 1)
            X.append(F[:, idx]); N.append(neg[idx])
            pm = np.zeros(W, bool)
            for h in hits:
                pm |= np.abs(idx / AF - h) <= POS_W
            P.append(pm)
        yield (torch.tensor(np.stack(X)), torch.tensor(np.stack(P)), torch.tensor(np.stack(N)))


def train():
    tr, te = split()
    t0 = time.time(); D = dataset(tr); nh = sum(len(d[2]) for d in D)
    print(f"train rallies {len(D)}, verified hits {nh}, negative frames {sum(d[3].sum() for d in D)}  ({time.time()-t0:.0f}s)", flush=True)
    net = Net().to(DEV); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-3)
    it = batches(D); bce = nn.BCEWithLogitsLoss()
    for step in range(3000):
        X, P, N = (t.to(DEV) for t in next(it))
        lg = net(X)
        # positives: the best frame inside each positive window should fire (max over the window)
        lp = []
        for b in range(len(lg)):
            m = P[b]
            if m.any():
                # contiguous windows: take the max per window
                idx = torch.nonzero(m)[:, 0]; br = torch.nonzero(torch.diff(idx) > 1)[:, 0] + 1
                for g in torch.tensor_split(idx, br.cpu()):
                    lp.append(lg[b, g].max())
        loss_p = bce(torch.stack(lp), torch.ones(len(lp), device=DEV)) if lp else 0
        loss_n = bce(lg[N], torch.zeros(int(N.sum()), device=DEV)) if N.any() else 0
        loss = loss_p + loss_n
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            print(f"step {step} loss {float(loss):.3f} (pos {float(loss_p):.3f} neg {float(loss_n):.3f})", flush=True)
    os.makedirs(OUT, exist_ok=True); torch.save(net.state_dict(), os.path.join(OUT, "model.pt"))


@torch.no_grad()
def prob(net, F):
    return torch.sigmoid(net(torch.tensor(F[None]).to(DEV)))[0].cpu().numpy()


def peaks(p, thr, dist=int(0.12 * AF)):
    from scipy.signal import find_peaks
    k, _ = find_peaks(p, height=thr, distance=dist); return k / AF


def evaluate():
    tr, te = split(); D = dataset(te)
    net = Net().to(DEV); net.load_state_dict(torch.load(os.path.join(OUT, "model.pt"))); net.eval()
    # baseline without learning: high-pass energy envelope peaks (same peak picking)
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 2000, "hp", fs=SR, output="sos")

    def env(u):
        a = np.load(os.path.join(AUD, u + ".npy")).astype(np.float32) / 32768.0
        y = sosfiltfilt(sos, a); e = np.sqrt(np.convolve(y * y, np.ones(HOP) / HOP, "same"))[::HOP]
        return e / (np.median(e) + 1e-9)
    rep = {}
    cands = [("learned", lambda u, F: prob(net, F), (0.3, 0.5, 0.7, 0.9))] + ([("highpass", lambda u, F: env(u), (4, 8))] if MODE == "audio" else [])
    for name, score, thrs in cands:
        for thr in thrs:
            tp = fp = fn = 0; dt = []; nsec = sum(d[3].sum() for d in D) / AF
            for u, F, hits, neg in D:
                p = score(u, F)[:F.shape[1]]
                pk = peaks(p, thr)
                # only judge where the teacher knows: near verified hits (+-0.08 s) or inside negative stretches
                t = pk; ii = np.clip((t * AF).astype(int), 0, len(neg) - 1)
                for h in hits:
                    d = np.abs(pk - h)
                    if len(d) and d.min() <= 0.06:
                        tp += 1; dt.append(float(pk[np.argmin(d)] - h))
                    else:
                        fn += 1
                fp += int(neg[ii].sum())
            rp = tp / max(tp + fn, 1); pp = tp / max(tp + fp, 1)
            rep[f"{name}@{thr}"] = dict(recall=rp, precision=pp, f1=2 * rp * pp / max(rp + pp, 1e-9),
                                        n_hits=tp + fn, false_in_flight=fp, timing_ms_med=1000 * float(np.median(np.abs(dt))) if dt else None)
            print(f"{name:9s} thr {thr:<4}: recall {rp:.2f} precision {pp:.2f} F1 {rep[f'{name}@{thr}']['f1']:.2f}  "
                  f"(hits {tp + fn}, false peaks in mid-flight {fp} = {fp / nsec:.2f}/s), |timing| median {rep[f'{name}@{thr}']['timing_ms_med']} ms", flush=True)
    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=1)


def probs():
    net = Net().to(DEV); net.load_state_dict(torch.load(os.path.join(OUT, "model.pt"))); net.eval()
    os.makedirs(os.path.join(OUT, "probs"), exist_ok=True)
    for p in sorted(glob.glob(os.path.join(AUD, "*.npy"))):
        u = os.path.basename(p)[:-4]
        np.save(os.path.join(OUT, "probs", u + ".npy"), prob(net, feats(u)).astype(np.float16))
    print("DONE", flush=True)


if __name__ == "__main__":
    {"train": train, "eval": evaluate, "probs": probs}[sys.argv[1]]()
