"""Train RallyNet (birdseye_v2/rallynet.py) on broadcast rallies without 3D labels; evaluate on DroneData against
two-view ground truth with the demo's metrics (tools/export_rallynet_viewer.py): player position vs the top-view
annotations, shuttle 3D error, labelled hits recovered within 3 frames.

Train sets: outputs/rally_obs/{bfmd,fb,fb20k}, plus the DroneData dev matches BD12-14 (hit times, two-drone triangulated
joints). The scored DroneData matches (BD15-23) are never trained on.
usage: python tools/train_rallynet.py [train|eval]   (env STEPS=20000, CROP=160, BATCH=12, TAG=v0, SETS=bfmd+fb+fb20k,
       ARCH=rallyphys, VTEACH=30; the released model is TAG=p11 with these defaults)"""
import glob, json, os, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from birdseye_v2 import rallynet as RN  # noqa: E402

OBS = os.path.join(ROOT, "outputs", "rally_obs")
OUT = os.path.join(ROOT, "outputs", "rallynet")
TAG = os.environ.get("TAG", "v0")
CROP = int(os.environ.get("CROP", "160"))
STEPS = int(os.environ.get("STEPS", "20000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "5000"))
BATCH = int(os.environ.get("BATCH", "12"))
SETS = os.environ.get("SETS", "bfmd+fb+fb20k").split("+")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ARCH = os.environ.get("ARCH", "rallyphys")  # rallyphys (birdseye_v2/rallyphys.py, the method) | rallynet (earlier direct-regression baseline)


def new_model():
    if ARCH == "rallyphys":
        from birdseye_v2 import rallyphys
        return rallyphys.RallyPhys().to(DEV)
    return RN.RallyNet().to(DEV)


def model_losses(out, b):
    if ARCH == "rallyphys":
        from birdseye_v2 import rallyphys
        return rallyphys.losses(out, b)
    return RN.losses(out, b)


def load_model(tag=None):
    m = new_model()
    m.load_state_dict(torch.load(os.path.join(OUT, (tag or TAG) + ".pt"), map_location=DEV), strict=ARCH == "rallynet")
    return m


_LAB = None


def labels():
    global _LAB
    if _LAB is None:
        lp = os.path.join(OBS, "hit_labels.json")
        _LAB = json.load(open(lp)) if os.path.exists(lp) else {"labels": {}, "bfmd_test": []}
    return _LAB


VTEACH = float(os.environ.get("VTEACH", "30"))  # weight of the VIMO bone-direction teacher (training only)
# COCO joints of our bones (from, to) and their SMPL-24 counterparts; the trunk is hip centre -> shoulder centre
TEACH_BONES = [(12, 11), (11, 13), (13, 15), (12, 14), (14, 16), None, (6, 5), (5, 7), (7, 9), (6, 8), (8, 10)]
_SMPL_OF = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}


def vimo_teacher(f, st, uid, agree=8.0):
    """VIMO's bone directions for each player slot -> Vd (T,P,11,3), Vd_mask (T,P,11). A bone counts only where both of
    its VIMO joints project within `agree` px (at f = 1000) of a confident YOLO keypoint: VIMO then only says how
    the bone runs in depth, which one view cannot."""
    T = len(f["C"])
    Vd = np.zeros((T, RN.P, len(TEACH_BONES), 3), np.float32)
    Vm = np.zeros((T, RN.P, len(TEACH_BONES)), np.float32)
    vp = os.path.join(ROOT, "outputs", st, "vimo_bc", uid + ".npz")
    if st not in ("bfmd", "fb", "fb20k") or not os.path.exists(vp):
        return Vd, Vm
    v = np.load(vp)
    fr = v["frames"].astype(int)
    keep = (fr >= 0) & (fr < T)
    fr = fr[keep]
    cj = sorted(_SMPL_OF)
    for who in ("top", "bottom"):
        X = v[who][keep]  # (n,24,3)
        ok = np.isfinite(X).all((1, 2)) & np.isfinite(f["C"][fr]).all(1)
        if not ok.any():
            continue
        X, frs = X[ok], fr[ok]
        foot = X[:, [7, 8], :2].mean(1)
        dd = np.where(f["present"][frs] > 0, np.linalg.norm(f["root"][frs] - foot[:, None], axis=-1), 1e9)
        q = dd.argmin(1)
        m = dd[np.arange(len(q)), q] < 0.8
        X, frs, q = X[m], frs[m], q[m]
        J = np.zeros((len(frs), 17, 3))
        for c in cj:
            J[:, c] = X[:, _SMPL_OF[c]]
        r = J[:, cj] - f["C"][frs][:, None]
        r /= np.linalg.norm(r, axis=-1, keepdims=True)
        e = np.linalg.norm(r - f["kp_dir"][frs, q][:, cj], axis=-1) * 1000
        good = np.zeros((len(frs), 17), bool)
        good[:, cj] = (e < agree) & (f["kp_conf"][frs, q][:, cj] > 0.5)
        for i, bc in enumerate(TEACH_BONES):
            if bc is None:
                a, b_ = J[:, [11, 12]].mean(1), J[:, [5, 6]].mean(1)
                g = good[:, [5, 6, 11, 12]].all(1)
            else:
                a, b_ = J[:, bc[0]], J[:, bc[1]]
                g = good[:, bc[0]] & good[:, bc[1]]
            d = b_ - a
            d /= np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9
            Vd[frs, q, i] = d
            Vm[frs, q, i] = g
    return Vd, Vm


def load_feats(path):
    """Model inputs + hit labels (T,P) where the rally's set has them (slot side from its median foot y)."""
    f = RN.featurize(dict(np.load(path)))
    T = len(f["C"])
    # each player's racket hand for the rally, voted by the racket detections on the flat first-guess body
    J0 = f["X0"] + np.concatenate([f["root"], np.zeros_like(f["root"][..., :1])], -1)[..., None, :]
    J0[f["kp_conf"] <= 0] = np.nan
    f["hand"] = np.repeat(RN.racket_hand(J0, f, return_vote=True)[1][None], T, 0)  # share of votes for the right hand
    st, uid = os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-4]
    lab = labels()["labels"].get(st, {}).get(uid)
    f["hit_lbl"] = np.zeros((T, RN.P), np.float32)
    f["hit_has"] = np.float32(lab is not None)
    f["hit_any"] = np.zeros(T, np.float32)
    f["hit_any_has"] = np.float32(0)
    if st == "drone" and uid.startswith(DEV_MATCHES):  # DroneData dev matches: hit frames, hitter not given
        for l in json.load(open(path[:-4] + "_gt.json"))["hits"]:
            if 0 <= int(round(l)) < T:
                f["hit_any"][int(round(l))] = 1.0
        f["hit_any_has"] = np.float32(1)
    f["J3d"] = np.zeros((T, RN.P, RN.J, 3), np.float32)
    f["J3d_mask"] = np.zeros((T, RN.P, RN.J), np.float32)
    if st == "drone" and uid.startswith(DEV_MATCHES):  # two-drone triangulated joints, matched to slots by position
        m_, r_ = uid.split("__")
        tp = os.path.join(ROOT, "outputs", "pose3d", m_, r_ + ".npz")
        if os.path.exists(tp):
            d = np.load(tp)
            for fr, Jt in zip(d["frames"], d["J"]):
                if not 0 <= fr < T:
                    continue
                c = np.nanmean(Jt[[11, 12, 15, 16], :2], 0)
                if not np.isfinite(c).all():
                    continue
                dist = np.where(f["present"][fr] > 0, np.linalg.norm(f["root"][fr] - c, axis=1), 1e9)
                q = int(np.argmin(dist))
                if dist[q] < 0.6:
                    ok = np.isfinite(Jt[:, 0])
                    f["J3d"][fr, q, ok] = Jt[ok]
                    f["J3d_mask"][fr, q] = ok
    if VTEACH > 0:
        f["Vd"], f["Vd_mask"] = vimo_teacher(f, st, uid)
    if lab is not None:
        from birdseye_v2.court import NET_Y
        main = {}  # singles: each side's label goes to that side's main player (most present), not officials
        for q in range(RN.P):
            ok = f["present"][:, q] > 0
            if ok.sum() < 5:
                continue
            side = "top" if np.median(f["root"][ok, q, 1]) > NET_Y else "bottom"
            if side not in main or ok.sum() > main[side][1]:
                main[side] = (q, ok.sum())
        for side, (q, _) in main.items():
            for fr in lab[side]:
                if 0 <= fr < T:
                    f["hit_lbl"][fr, q] = 1.0
    return f


class Crops(Dataset):
    def __init__(self, paths, n):
        self.paths, self.n = paths, n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng()
        for _ in range(20):
            f = load_feats(self.paths[rng.integers(len(self.paths))])
            T = len(f["C"])
            if T < 40 or f["present"].sum() < 20:
                continue
            a = int(rng.integers(0, max(1, T - CROP + 1)))
            return crop(f, a, a + CROP)
        return crop(f, 0, CROP)


def crop(f, a, b):
    out = {}
    for k, v in f.items():
        if np.ndim(v) == 0:
            out[k] = np.float32(v)
            continue
        x = v[a:b]
        if len(x) < b - a:  # pad (masked: cam_ok / present / vis = 0)
            x = np.concatenate([x, np.zeros((b - a - len(x),) + x.shape[1:], x.dtype)])
        out[k] = x
    return out


def to_dev(b):
    return {k: (torch.as_tensor(v) if not torch.is_tensor(v) else v).to(DEV).float() for k, v in b.items()}


def train():
    test = set(labels()["bfmd_test"])  # BFMD's test matches: never trained on
    paths = [p for s in SETS for p in glob.glob(os.path.join(OBS, s, "*.npz")) if os.path.basename(p)[:-4] not in test]
    if os.environ.get("DRONE_DEV", "1") == "1":  # DroneData dev matches (BD12-14) with their hit labels, x3
        paths += [p for p in glob.glob(os.path.join(OBS, "drone", "*.npz")) if os.path.basename(p).startswith(DEV_MATCHES)] * 3
    print(len(paths), "training rallies from", SETS, flush=True)
    model = new_model()
    print(ARCH, sum(p.numel() for p in model.parameters()) / 1e6, "M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=0.05)
    dl = DataLoader(Crops(paths, STEPS * BATCH), batch_size=BATCH, num_workers=3, persistent_workers=True)
    os.makedirs(OUT, exist_ok=True)
    t0, run = time.time(), {}
    for step, b in enumerate(dl):
        b = to_dev(b)
        if not all(torch.isfinite(v).all() for v in b.values()):
            print("skip: non-finite input", flush=True)
            continue
        if ARCH == "rallyphys":  # chain the shuttle's flights at the labelled hits where a rally has them (70%)
            lab = torch.maximum(b["hit_lbl"].max(-1).values, b["hit_any"])
            has = ((b["hit_has"] > 0) | (b["hit_any_has"] > 0)) & (torch.rand(len(lab), device=DEV) < 0.7)
            out = model(b, hits=lab, has=has)
        else:
            out = model(b)
        loss, parts = model_losses(out, b)
        if not torch.isfinite(loss):
            print("skip: non-finite loss", {k: v for k, v in parts.items() if not np.isfinite(v)}, flush=True)
            continue
        opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            print("skip: non-finite gradient", flush=True)
            opt.zero_grad()
            continue
        opt.step(); sched.step()
        for k, v in parts.items():
            run[k] = 0.98 * run.get(k, v) + 0.02 * v
        if step % 200 == 0:
            print(f"{step} {time.time() - t0:.0f}s " + " ".join(f"{k} {v:.3g}" for k, v in run.items()), flush=True)
        if step % 2000 == 0 or step == STEPS - 1:
            torch.save(model.state_dict(), os.path.join(OUT, TAG + ".pt"))
        if step and (step % EVAL_EVERY == 0 or step == STEPS - 1):
            global PHYS
            PHYS = step == STEPS - 1 and ARCH == "rallynet"  # the physics layer (slow: one fit per shot) only at the end
            print("EVAL", step, "physics" if PHYS else "net-shuttle", evaluate(model), flush=True)
            model.train()
        if step >= STEPS - 1:
            break


@torch.no_grad()
def infer(model, f, win=CROP, stride=CROP // 2):
    """Whole rally by overlapping windows, outputs averaged with a triangular weight."""
    model.eval()
    T = len(f["C"])
    acc, wsum = None, np.zeros(T)
    starts = list(range(0, max(1, T - win + 1), stride))
    if starts[-1] + win < T:
        starts.append(T - win)
    for a in starts:
        b = to_dev({k: v[None] for k, v in crop(f, a, a + win).items()})
        out = {k: v[0].cpu().numpy() for k, v in model(b).items() if v.shape[1] == win}  # per-frame outputs
        n = min(win, T - a)
        w = np.minimum(np.arange(1, n + 1), np.arange(n, 0, -1)).astype(float)
        if acc is None:
            acc = {k: np.zeros((T,) + v.shape[1:]) for k, v in out.items()}
        for k, v in out.items():
            acc[k][a:a + n] += v[:n] * w.reshape(-1, *[1] * (v.ndim - 1))
        wsum[a:a + n] += w
    return {k: v / wsum.reshape(-1, *[1] * (v.ndim - 1)) for k, v in acc.items()}


def hits_from(prob, thr=None, nms=6):
    thr = HIT_THR if thr is None else thr
    from scipy.signal import find_peaks
    pk, _ = find_peaks(prob, height=thr, distance=nms)
    return pk


DEV_MATCHES = ("BD12_", "BD13_", "BD14_")  # DroneData dev matches (thresholds are chosen here); never in the scores
HIT_THR = float(os.environ.get("HIT_THR", "0.5"))
DEMO = ["BD15__Rally-39", "BD15__Rally-31", "BD17__Rally-1", "BD19__Rally-32", "BD21__Rally-9", "BD23__Rally-27"]


def eval_paths(full=False):
    """All DroneData rallies, or a fixed 60 (the six demo rallies + 54 others) for checks during training."""
    ps = [p for p in sorted(glob.glob(os.path.join(OBS, "drone", "*.npz"))) if not os.path.basename(p).startswith(DEV_MATCHES)]
    if full:
        return ps
    demo = [p for p in ps if os.path.basename(p)[:-4] in DEMO]
    rest = [p for p in ps if p not in demo]
    return demo + [rest[i] for i in np.random.default_rng(0).choice(len(rest), min(54, len(rest)), replace=False)]


PHYS = os.environ.get("PHYS", "1") == "1" and ARCH == "rallynet"  # RallyPhys integrates its own shuttle
CHAIN_SHOTS = os.environ.get("CHAIN_SHOTS", "0") == "1"  # chain contacts at the per-shot fit's shots (else the hits)
REFINE = os.environ.get("REFINE", "1") == "1"  # add the network's sub-threshold hits where one flight cannot explain a shot
REFINE_PX = 20.0  # ... only when some shot's flight misses its detections by more than this (median px)
MIN_SHOT = 5  # frames: an added hit is at least this far from the shot's ends
SERVE_CLAMP = os.environ.get("SERVE_CLAMP", "1") == "1"  # hits before the track starts move to its start (no gap to fill)
SIDE_VOTE = os.environ.get("SIDE_VOTE", "1") == "1"  # the network's hitters vote the serve side (shuttle3d.fit_rally)


def physics_shuttle(p, f, out, return_shots=False):
    """The physics output layer: one gravity + drag flight per shot (birdseye_v2.pipeline.reconstruct_rally),
    split at the network's hits and started at the network's racket head of the player whose hand is nearest the
    shuttle in the image. -> (T,3) shuttle positions."""
    from birdseye_v2 import pipeline as PL
    from birdseye_v2.court import Camera, NET_Y, L, W
    gt = json.load(open(p[:-4] + "_gt.json"))
    cam = Camera.from_dict(gt["cam"])
    m, r = os.path.basename(p)[:-4].split("__")
    g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
    fr = np.array(g["back_track"]["frames"], float)
    px = np.array(g["back_track"]["px"], float).reshape(-1, 2)
    T = len(f["C"])
    jt, u, pres = out["joints"], out["racket_u"], f["present"] > 0
    if "u2" in out:  # RallyPhys: the network's hand, one per player for the whole rally
        hand = (np.nanmean(np.where(pres, out["hand_p"], np.nan), 0) > 0.5).astype(int)
        u = out["u2"][:, np.arange(RN.P), hand]
    else:
        hand = RN.racket_hand(jt, f)  # one racket hand per player for the whole rally
    wrist = jt[:, np.arange(jt.shape[1]), 9 + hand]  # (T,P,3)
    sp = 1 / (1 + np.exp(-out["hit"])) * f["present"]  # (T,P) per-player hit probability
    far = jt[:, :, [15, 16], 1].mean(2) > NET_Y

    def cands(cam_, poses, frames, pxs, t, rackets=None):
        k = int(round(t))
        if not 0 <= k < T:
            return []
        uv = PL._interp(frames, pxs, t)
        res = []
        # the network's hitter: P(far side) from the best per-player hit probability on each side near the hit
        w = sp[max(0, k - 2):k + 3].max(0)
        pf, pn = (w[far[k] & pres[k]].max(initial=0), w[~far[k] & pres[k]].max(initial=0))
        far_p = float(pf / (pf + pn)) if pf + pn > 0.2 else None
        for q in np.where(pres[k])[0]:
            h = wrist[k, q] + RN.GRIP_TO_HEAD * u[k, q]
            if not np.isfinite(h).all():
                continue
            d = np.linalg.norm(cam_.project(h[None])[0] - uv)
            res.append({"hand": h, "side": int(jt[k, q, [15, 16], 1].mean() > NET_Y), "px": d,
                        **({"far_p": far_p} if SIDE_VOTE else {})})
        return sorted(res, key=lambda c: c["px"])
    hp = 1 - np.prod(1 - sp, axis=1)
    hits = [float(h) for h in (hits_from(hp, nms=8) if "u2" in out else hits_from(hp))]
    if SERVE_CLAMP:  # a serve hit before the track starts: in flight at its start
        hits = sorted(set(max(h, float(fr[0])) for h in hits))
    if not hits or hits[0] > fr[0] + 4:
        hits = [float(fr[0])] + hits
    old = PL._candidates
    def shot_px(rec):  # each shot's median reprojection error (px): does one flight explain it?
        fit, w = rec["fit"], []
        for fr_, uv in fit.obs:
            w.append(float(np.median(np.linalg.norm(cam.project(fit.position(np.asarray(fr_, float))) - uv, axis=1)))
                     if len(fr_) >= 3 else 0.0)
        return w

    def inside(rec):  # a fit that leaves the hall has flown off along its rays
        X = rec["fit"].position(np.arange(int(fr[0]), int(fr[-1]) + 1, dtype=float))
        X = X[np.isfinite(X).all(1)]
        return bool(np.all((X[:, 0] > -3) & (X[:, 0] < W + 3) & (X[:, 1] > -3) & (X[:, 1] < L + 3) & (X[:, 2] < 12)))
    PL._candidates = cands
    try:
        rec = PL.reconstruct_rally(cam, fr, px, [np.zeros((0, 17, 3))] * T, hits=hits)
        # a shot one flight cannot explain hides a hit the network missed: add one at the shot's strongest
        # sub-threshold hit peak, keep it if the fit explains the track better and stays in the hall
        for _ in range(4 if REFINE else 0):
            w = shot_px(rec)
            add = []
            for (a, b), e in zip(rec["fit"].shots, w):
                lo, hi = int(a) + MIN_SHOT, int(b) - MIN_SHOT
                if e > REFINE_PX and hi > lo and hp[lo:hi].max() > 0.05:
                    add.append(float(lo + np.argmax(hp[lo:hi])))
            if not add:
                break
            try:
                rec2 = PL.reconstruct_rally(cam, fr, px, [np.zeros((0, 17, 3))] * T, hits=sorted(hits + add))
            except ValueError:
                break
            if not inside(rec2) or np.minimum(shot_px(rec2), 200).sum() >= np.minimum(w, 200).sum():
                break
            rec, hits = rec2, sorted(hits + add)
    except ValueError:  # a degenerate fit: keep the network's own shuttle
        return (out["shuttle"], None) if return_shots else out["shuttle"]
    finally:
        PL._candidates = old
    S = rec["fit"].position(np.arange(T, dtype=float), rest_until=T - 1)
    ok = np.isfinite(S).all(1)
    if ok.any():  # outside the fitted shots the shuttle holds its nearest fitted position (no network fallback)
        idx = np.where(ok)[0]
        S = S[idx[np.clip(np.searchsorted(idx, np.arange(T)), 0, len(idx) - 1)]] if not ok.all() else S
        S = np.where(ok[:, None], rec["fit"].position(np.arange(T, dtype=float), rest_until=T - 1), S)
    else:
        S = out["shuttle"]
    return (S, [tuple(map(float, s)) for s in rec["fit"].shots]) if return_shots else S


def rallyphys_shuttle(p, f, out):
    """RallyPhys's shuttle: the hitter-anchored per-shot fit, then one exact continuous chain through its shots."""
    from birdseye_v2 import rallyphys
    hp = 1 - np.prod(1 - 1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    S0, shots = physics_shuttle(p, f, out, return_shots=True)
    return rallyphys.chain_refit(S0, hits_from(hp, nms=8), f, shots=shots if CHAIN_SHOTS else None)


def evaluate_one(model, p, out=None, S=None):
    """One DroneData rally: median player error (ankle midpoint vs top-view annotation), shuttle 3D error at the
    two-view GT times, labelled hits recovered within 3 frames."""
    from eval_players import match_err
    gt = json.load(open(p[:-4] + "_gt.json"))
    f = load_feats(p)
    out = out if out is not None else infer(model, f)
    T = len(f["C"])
    feet = out["joints"][:, :, [15, 16], :2].mean(2)
    pe = []
    for fr, P in gt["players"].items():
        fr = int(fr)
        if 0 <= fr < T:
            pr = feet[fr][f["present"][fr] > 0]
            pe += [x for x in match_err(np.array(P), pr) if x is not None]
    if S is not None:
        pass
    elif PHYS:
        S = physics_shuttle(p, f, out)
    elif ARCH == "rallyphys":  # per-shot fit through the network's contacts, then one exact continuous chain
        S = rallyphys_shuttle(p, f, out)
    else:
        S = out["shuttle"]
    se = []
    for t, *X in gt["shuttle"]:
        k = int(np.floor(t))
        if 0 <= k < T - 1:
            x = S[k] + (t - k) * (S[k + 1] - S[k])
            se.append(np.linalg.norm(x - np.array(X)))
    hp = 1 - np.prod(1 - 1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    hits = hits_from(hp, nms=8) if ARCH == "rallyphys" else hits_from(hp)
    rec = float(np.mean([any(abs(h - l) <= 3 for h in hits) for l in gt["hits"]])) if gt["hits"] else np.nan
    r3 = lambda v: round(float(v), 3)
    return {"player": r3(np.median(pe)) if pe else None, "shuttle": r3(np.median(se)) if se else None,
            "shuttle_mean": r3(np.mean(se)) if se else None, "n_gt": len(se), "recall": round(rec, 2),
            "hits": len(hits), "labels": len(gt["hits"])}


def evaluate(model, verbose=False, full=False):
    res = []
    for p in eval_paths(full):
        e = evaluate_one(model, p)
        res.append((e["player"] if e["player"] is not None else np.nan, e["shuttle"] if e["shuttle"] is not None else np.nan,
                    e["recall"], e["hits"], e["labels"]))
        if verbose:
            print(os.path.basename(p), "player %.3f shuttle %.3f recall %.2f hits %d/%d" % res[-1], flush=True)
    r = np.array(res, float)
    return {"rallies": len(r), "player_med": round(float(np.nanmedian(r[:, 0])), 3), "shuttle_med": round(float(np.nanmedian(r[:, 1])), 3),
            "hit_recall": round(float(np.nanmean(r[:, 2])), 3), "hits_per_label": round(float(r[:, 3].sum() / max(r[:, 4].sum(), 1)), 2)}


if __name__ == "__main__":
    if (sys.argv[1:] or ["train"])[0] == "train":
        train()
    else:
        m = load_model()
        print(evaluate(m, verbose=True, full=os.environ.get("FULL") == "1"))
