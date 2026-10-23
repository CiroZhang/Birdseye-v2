"""Birdseye v1's BFMD hit + stroke-type model (third_party/birdseye_v1/action/train_final.py), trained and scored
on any v1-layout training set -- v1's own (data/bfmd/v1) or ours (data/bfmd/v2, tools/bfmd_export.py) -- with
v1's split, recipe and metric unchanged. Only the device (CUDA) and the checkpoint path differ.

v1's protocol picks the epoch and the peak threshold on the same 2 validation matches it reports. With
PROTOCOL=clean the 10 training matches are split again by the same rule (seed 0): epoch/threshold are picked
on 2 of them, the model is trained on the other 8, and the 2 v1 validation matches are scored once.
FEAT=shuttle (or shuttle+racket, racket) adds the TrackNet shuttle (outputs/bfmd/refined) as two extra "joints" per frame: where its camera
ray meets the floor and the plane 2 m up (together they fix the ray), pelvis-relative and canonicalised like v1's
joints, so v1's velocity/acceleration features and left-right mirroring apply unchanged (in_dim 360 -> 390).
FEAT=...racket adds each player's racket head (tools/bfmd_racket.py boxes, held in the VIMO hand as in the viewer:
tools/fb_full_example.rackets), only while a racket box was seen near that hand, as one more point (+15 dims).
Stroke-head options (off by default = v1): SHOT_SPREAD=k also labels the k frames either side of a hit with its
stroke type (training only; v1 labels the hit frame alone), SHOT_LS label smoothing, SHOT_W the stroke-loss weight
(v1 0.3), SELECT=f1acc picks the epoch on hit F1 + stroke accuracy instead of hit F1.
REFIT=1 (clean protocol): after the epoch and threshold are picked on the 2 selection matches, train a fresh model
on all 10 non-test matches for that many epochs (same schedule) and score that one on the test matches.
usage: BFMD_DATA_ROOT=data/bfmd/v2 python tools/bfmd_stroke.py   (env PROTOCOL=v1|clean, FEAT=pose|shuttle, SEED, TAG, EPOCHS=60)"""
import json, os, sys, types

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party", "birdseye_v1", "action"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_final as tf  # noqa: E402
from hit_model_transformer import HitSpotterTransformer  # noqa: E402

PROTOCOL = os.environ.get("PROTOCOL", "v1")
FEAT = os.environ.get("FEAT", "pose")
FEATS = set(FEAT.split("+"))
SEED = os.environ.get("SEED")  # v1 sets none; set for repeat runs
SHOT_SPREAD = int(os.environ.get("SHOT_SPREAD", "0"))
SHOT_LS = float(os.environ.get("SHOT_LS", "0"))
SHOT_W = float(os.environ.get("SHOT_W", "0.3"))
SELECT = os.environ.get("SELECT", "f1")
REFIT = os.environ.get("REFIT") == "1"
HEAD = "".join(f"_{k}{v}" for k, v, d in (("sp", SHOT_SPREAD, 0), ("ls", SHOT_LS, 0), ("w", SHOT_W, 0.3), ("sel", SELECT, "f1")) if v != d) + ("_refit" if REFIT else "")
TAG = os.environ.get("TAG", os.path.basename(os.path.normpath(tf.BASE)) + "_" + PROTOCOL + ("" if FEAT == "pose" else "_" + FEAT) + HEAD + ("" if SEED is None else "_s" + SEED))
N_EPOCHS = int(os.environ.get("EPOCHS", "60"))
OUT = os.path.join(ROOT, "outputs", "bfmd", "stroke")
tf.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def split_matches(match_of_rally, matches):
    """v1's load_split rule applied to a subset of matches."""
    matches = sorted(matches)
    perm = np.random.default_rng(0).permutation(len(matches))
    n_val = max(2, len(matches) // 5)
    return {matches[i] for i in perm[n_val:]}, {matches[i] for i in perm[:n_val]}


SHUTTLE_Z = (0.0, 2.0)  # m: the two planes the shuttle's camera ray is cut with
# where the per-rally inputs live; tools/fbbench_answer.py points these at the FBBench clips
SRC = {"sd": os.path.join(ROOT, "data", "bfmd", os.environ.get("SD", "v2_sd")),  # <sd>/<match>/joints_selfdetected.json
       "out": os.path.join(ROOT, "outputs", "bfmd"),  # <out>/{calib_bc,refined,racket}/<uid>.*
       "uid": lambda m, rt: f"{m}__{rt}"}


def set_pelvis(dataset):
    """World pelvis per side (from the self-detected joints) for append_points."""
    sd = SRC["sd"]
    cache = {}
    for tag, r in dataset["rallies"].items():
        m, rt = tag.split("::")
        if m not in cache:
            cache[m] = json.load(open(os.path.join(sd, m, "joints_selfdetected.json")))["rallies"]
        n = r["n_frames"]
        for side in ("top", "bottom"):
            J = cache[m][rt]["joints"].get(side)
            P = np.full((n, 3), np.nan)
            if J is not None:
                J = np.array(J, float)[:n, 0]
                P[:len(J)] = J
            r["_pelvis_" + side] = P


def append_points(r, side, pts):
    """Append (T,k,3) world points to one side's _rel/_vel as extra joints, pelvis-relative and canonicalised."""
    n = r["n_frames"]
    pelvis = r["_pelvis_" + side]
    rel = np.clip(pts - pelvis[:, None], -20, 20)
    if side == "top":  # as build_training_data_selfdetected.canonicalize
        rel[..., :2] *= -1
    vel = np.zeros_like(rel)
    vel[1:] = rel[1:] - rel[:-1]
    vel[~np.isfinite(vel)] = 0  # no velocity across a missing point
    r[side + "_rel"] = np.concatenate([np.array(r[side + "_rel"], np.float32), rel], 1)
    r[side + "_vel"] = np.concatenate([np.array(r[side + "_vel"], np.float32), vel], 1)


def add_racket(dataset):
    """Append each side's racket head (T,1,3) to its own _rel/_vel, in place."""
    from fb_full_example import HEAD_R, rackets
    from birdseye_v2.court import Camera
    from birdseye_v2.players import load_rackets
    sd = SRC["sd"]
    joints_by_match, n_seen, n_all = {}, 0, 0
    for tag, r in dataset["rallies"].items():
        m, rt = tag.split("::")
        if m not in joints_by_match:
            joints_by_match[m] = json.load(open(os.path.join(sd, m, "joints_selfdetected.json")))["rallies"]
        uid = SRC["uid"](m, rt)
        n = r["n_frames"]
        cam = frame_cams(uid, n)
        p = os.path.join(SRC["out"], "racket", uid + ".npz")
        boxes = load_rackets(p) if os.path.exists(p) else []
        boxes = boxes + [np.zeros((0, 5))] * max(0, n - len(boxes))
        for side in ("top", "bottom"):
            head = np.full((n, 1, 3), np.nan)
            J = joints_by_match[m][rt]["joints"].get(side)
            if J is not None:
                J = np.array(J, float)[:n]
                J = np.concatenate([J, np.full((n - len(J), 24, 3), np.nan)]) if len(J) < n else J
                if np.isfinite(J[:, 0, 0]).sum() >= 8:
                    w, u, near, _ = rackets(cam, J, boxes[:n])
                    h = w + HEAD_R * u
                    h[~near] = np.nan
                    head[:, 0] = h
                    n_seen += near.sum()
            n_all += n
            append_points(r, side, head)
    print(f"racket seen in {n_seen / max(n_all, 1):.3f} of player-frames", flush=True)


CAMTRACK = os.environ.get("CAMTRACK", "1") == "1"


def video_size(uid):
    import cv2
    for d in (("fbbench", "hf_video_clips_qa"), ("bfmd", "clips")):
        p = os.path.join(ROOT, "data", *d, uid + ".mp4")
        if os.path.exists(p):
            c = cv2.VideoCapture(p)
            return int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return None


def frame_cams(uid, n):
    """The clip camera, or per frame (list, None = no court view) from <out>/camtrack/<uid>.json when present
    (birdseye_v2/camtrack.py). None when the clip has no calibration."""
    from birdseye_v2 import camtrack
    from birdseye_v2.court import Camera
    p = os.path.join(SRC["out"], "calib_bc", uid + ".json")
    c = json.load(open(p)).get("cam") if os.path.exists(p) else None
    cam = Camera.from_dict(c) if c else None
    tp = os.path.join(SRC["out"], "camtrack", uid + ".json")
    if CAMTRACK and os.path.exists(tp):
        size = cam.size if cam is not None else video_size(uid)
        cams = camtrack.load(tp, size)[:n] if size else []
        if any(x is not None for x in cams):
            return cams + [None] * (n - len(cams))
    return cam


def add_shuttle(dataset):
    """Append the shuttle's two ray points to every rally's top/bottom _rel (T,24,3)->(T,26,3) and _vel, in place."""
    from birdseye_v2.court import Camera
    sd = SRC["sd"]
    by_match = {}
    for tag in dataset["rallies"]:
        by_match.setdefault(tag.split("::")[0], []).append(tag)
    n_vis = n_all = 0
    for m, tags in by_match.items():
        joints = json.load(open(os.path.join(sd, m, "joints_selfdetected.json")))["rallies"]
        for tag in tags:
            r, rt = dataset["rallies"][tag], tag.split("::")[1]
            uid = SRC["uid"](m, rt)
            n = r["n_frames"]
            ray = np.full((n, 2, 3), np.nan)
            p = os.path.join(SRC["out"], "refined", uid + ".json")
            cam = frame_cams(uid, n) if os.path.exists(p) else None
            if cam is not None:  # no valid court calibration: shuttle points stay NaN
                tr = json.load(open(p))
                f = np.array(tr["frames"], int)
                k = f < n
                if isinstance(cam, list):
                    k &= np.array([fi < n and cam[fi] is not None for fi in f], bool)
                if k.any():
                    px = np.array(tr["px"], float).reshape(-1, 2)[k]
                    for j, z in enumerate(SHUTTLE_Z):
                        if isinstance(cam, list):
                            g = np.stack([cam[fi].pixel_to_ground(q[None], z=z)[0] for fi, q in zip(f[k], px)])
                        else:
                            g = cam.pixel_to_ground(px, z=z)
                        ray[f[k], j] = np.c_[g[:, :2], np.full(len(g), z)]
            vis = np.isfinite(ray[:, 0, 0])
            n_vis += vis.sum(); n_all += n
            for side in ("top", "bottom"):
                append_points(r, side, ray)  # near-horizontal rays meet the floor far away: clipped
    tf.LR_PERM = tf.LR_PERM + [len(tf.LR_PERM), len(tf.LR_PERM) + 1]  # no left/right twin; mirror_lr only flips x
    print(f"shuttle visible in {n_vis / max(n_all, 1):.3f} of frames", flush=True)


def patch_shot_loss():
    """run_epoch computes hit + 0.3 * F.cross_entropy(shot); swap in smoothing and the weight via its F."""
    F = tf.F
    def ce(logits, target, **kw):
        return F.cross_entropy(logits, target, label_smoothing=SHOT_LS, **kw) * (SHOT_W / 0.3)
    tf.F = types.SimpleNamespace(**{k: getattr(F, k) for k in dir(F) if not k.startswith("__")})
    tf.F.cross_entropy = ce


def spread_shots(ds, k):
    """Give the k frames either side of each labelled hit its stroke type (nearest hit wins)."""
    for i, s in enumerate(ds.samples):
        shot = s[6]
        hits = np.flatnonzero(shot >= 0)
        if not len(hits):
            continue
        t = np.arange(len(shot))
        near = hits[np.abs(t[:, None] - hits[None]).argmin(1)]
        new = np.where(np.abs(t - near) <= k, shot[near], -1)
        ds.samples[i] = s[:6] + (new,)


def best_threshold(model, ds):
    best = (-1, 0.3)
    for t in THRESHOLDS:
        f1 = tf.evaluate_peaks(model, ds, t)[2]
        if f1 > best[0]:
            best = (f1, t)
    return best[1]


def setup(train_ds, in_dim, n_shot):
    """v1's model, optimiser, cosine schedule and positive weight for one training run."""
    tgt = np.concatenate([s[5] for s in train_ds.samples])
    pos_frac = (tgt > 0.5).mean()
    pos_weight = min(10.0, (1 - pos_frac) / max(pos_frac, 1e-4))
    loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=tf.collate)
    model = HitSpotterTransformer(in_dim=in_dim, d_model=192, n_heads=6, n_layers=4, n_shot_types=n_shot).to(tf.DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS)
    return model, opt, sched, loader, pos_weight


def main():
    if SEED is not None:
        torch.manual_seed(int(SEED)); np.random.seed(int(SEED))
    dataset = json.load(open(f"{tf.BASE}/training_dataset_all_selfdetected.json"))
    set_pelvis(dataset)
    if "shuttle" in FEATS:
        add_shuttle(dataset)
    if "racket" in FEATS:
        add_racket(dataset)
        tf.LR_PERM = tf.LR_PERM + [len(tf.LR_PERM)]
    vocab = dataset["shot_type_vocab"]
    shot_to_idx = {s: i for i, s in enumerate(vocab)}
    _, test_tags, test_matches = tf.load_split(dataset)
    mor = dataset["match_of_rally"]
    if PROTOCOL == "clean":
        tr_m, sel_m = split_matches(mor, set(mor.values()) - test_matches)
    else:  # v1: select on the reported matches
        tr_m, sel_m = set(mor.values()) - test_matches, test_matches
    tags = sorted(dataset["rallies"])
    train_ds = tf.RallyDataset(dataset, [t for t in tags if mor[t] in tr_m], shot_to_idx, augment=True)
    sel_ds = tf.RallyDataset(dataset, [t for t in tags if mor[t] in sel_m], shot_to_idx, augment=False)
    test_ds = tf.RallyDataset(dataset, test_tags, shot_to_idx, augment=False)
    print(f"{TAG}: train {len(train_ds)} sel {len(sel_ds)} test {len(test_ds)} samples; test {sorted(test_matches)}", flush=True)

    if SHOT_SPREAD:
        spread_shots(train_ds, SHOT_SPREAD)
    if SHOT_LS or SHOT_W != 0.3:
        patch_shot_loss()
    in_dim = tf.make_features(*train_ds.samples[0][:5]).shape[1]  # 360 = v1
    model, opt, sched, loader, pos_weight = setup(train_ds, in_dim, len(vocab))
    os.makedirs(OUT, exist_ok=True)
    ckpt = os.path.join(OUT, TAG + ".pt")
    best_f1, log = -1, []
    for epoch in range(N_EPOCHS):
        tr_hit, tr_shot = tf.run_epoch(model, loader, opt, pos_weight=pos_weight)
        sched.step()
        if epoch % 5 == 0 or epoch == N_EPOCHS - 1:  # v1 checks every 5 epochs
            t = best_threshold(model, sel_ds)
            p, r, f1, sacc, tp, fp, fn = tf.evaluate_peaks(model, sel_ds, t)
            print(f"epoch {epoch:2d}: train hit={tr_hit:.4f} shot={tr_shot:.3f} | sel [t={t}] P={p:.3f} R={r:.3f} "
                  f"F1={f1:.3f} shot_acc={sacc:.3f}", flush=True)
            log.append({"epoch": epoch, "thresh": t, "P": p, "R": r, "F1": f1, "shot_acc": sacc})
            score = f1 + sacc if SELECT == "f1acc" else f1
            if score > best_f1:
                best_f1, best_t, best_epoch = score, t, epoch
                torch.save(model.state_dict(), ckpt)
    model.load_state_dict(torch.load(ckpt, map_location=tf.DEVICE))
    if REFIT and PROTOCOL == "clean":
        full_ds = tf.RallyDataset(dataset, [t for t in tags if mor[t] in tr_m | sel_m], shot_to_idx, augment=True)
        if SHOT_SPREAD:
            spread_shots(full_ds, SHOT_SPREAD)
        model, opt, sched, loader, pos_weight = setup(full_ds, in_dim, len(vocab))
        for epoch in range(best_epoch + 1):
            tf.run_epoch(model, loader, opt, pos_weight=pos_weight)
            sched.step()
        torch.save(model.state_dict(), ckpt)
        print(f"refit on {len(full_ds)} samples for {best_epoch + 1} epochs", flush=True)
    p, r, f1, sacc, tp, fp, fn = tf.evaluate_peaks(model, test_ds, best_t)
    res = {"tag": TAG, "protocol": PROTOCOL, "data": tf.BASE, "feat": FEAT, "thresh": best_t, "epoch": best_epoch, "P": p, "R": r, "F1": f1,
           "shot_acc": sacc, "tp": tp, "fp": fp, "fn": fn, "test_matches": sorted(test_matches), "log": log}
    json.dump(res, open(os.path.join(OUT, TAG + ".json"), "w"), indent=1)
    print(f"RESULT {TAG}: test P={p:.3f} R={r:.3f} F1={f1:.3f} shot_acc={sacc:.3f} (tp={tp} fp={fp} fn={fn})", flush=True)


if __name__ == "__main__":
    main()
