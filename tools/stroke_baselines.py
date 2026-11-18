"""Published skeleton stroke classifiers -- BST (Chang, CVPRW 2026), TemPose-TF (Ibh et al. 2023, as re-implemented in
the BST repository) and ST-GCN -- retrained on exactly the hits, labels and match split of tools/stroke3d.py, with the
inputs and recipe of the BST repository (third_party/bst, github.com/Va6lue/BST-Badminton-Stroke-type-Transformer).

Inputs per labelled hit, as BST's prepare_train_on_shuttleset.py (2D only, no 3D):
  joints   COCO-17 YOLOv8-pose keypoints of both players (the same tracked detections as our 2D row,
           tools/bfmd_export_2d.side_kp), relative to each player's keypoint box centre and divided by its diagonal
  pos      each player's mid-ankle on the court plane (court calibration), divided by the court size -> [0, 1]
  shuttle  the TrackNet shuttle pixel (outputs/<set>/refined) divided by the video size -> [0, 1]
Player 0 is the hitter, player 1 the opponent (BST instead folds the side into the class). The clip is the window of
+-W frames around the hit cut to the rally, left-aligned, with its length as BST's video_len.
Recipe (BST repo): AdamW, cosine schedule with 400 warm-up steps, batch 128, label smoothing 0.1, random translation
of the joints, at most 1600 epochs, early stop after 300 without a better selection macro-F1, model at the best one.

STAGE=build writes the clips (data/<set>/bst_clips_W<W>.npz); STAGE=train MODEL=bst|tempose|stgcn trains SEEDS.
env: BFMD_DATA_ROOT / KEEP / SD as tools/stroke3d.py (they fix the hits), W (default 15), SET=bfmd|fbs"""
import json, os, sys, time, copy

os.environ.setdefault("PROTOCOL", "clean")
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "third_party", "bst", "stroke_classification"))
W = int(os.environ.get("W", "15"))
SET = os.environ.get("SET", "bfmd")
REP = os.environ.get("REP", "2d")  # 2d: BST's own inputs; 3d: the same network on Birdseye's 3D bodies, racket, shuttle
CLIPS = os.path.join(ROOT, "data", "bfmd" if SET == "bfmd" else "fbs",
                     f"bst_clips{'' if REP == '2d' else REP}_W{W}.npz")
BODY_SD = os.environ.get("BODY_SD", "")  # REP=3d: take the bodies from this <sd> dir instead
HP = json.loads(os.environ.get("HP", "{}"))  # BST size/dropout overrides (d_model, drop_p, ...), picked on selection
XTEST = os.environ.get("XTEST", "")  # path of the other dataset's clips (same REP) for cross-broadcast testing
# BFMD and FineBadminton-20K stroke names -> their shared classes. FB's kill subtypes are all smashes; FB's push shot
# is mostly "flat lift", so push and lift are one class in both; BFMD press has no FB twin (dropped)
COMMON = {"block": 0, "clear": 1, "drive": 2, "drop": 3, "drop shot": 3, "lift": 4, "net lift": 4, "push": 4,
          "push shot": 4, "net_kill": 5, "net kill": 5, "net_shot": 6, "net shot": 6, "cross-court net shot": 6,
          "serve": 7, "flick_serve": 7, "smash": 8, "kill": 8}
SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
BONES3D = [(p, c) for c, p in enumerate(SMPL_PARENTS) if p >= 0] + [(20, 24), (21, 24)]  # + racket head to wrists
OUT = os.path.join(ROOT, "outputs", "bfmd" if SET == "bfmd" else "fbs", "stroke_baselines")


def at_30fps(A, frames, n):
    """(T_native, ...) at kept indices `frames` -> (n, ...), each frame from the last kept one (bfmd_export.to_30fps)."""
    src = np.searchsorted(frames, np.arange(n), side="right") - 1
    out = np.full((n,) + A.shape[1:], np.nan)
    ok = (src >= 0) & (src < len(A))
    out[ok] = A[src[ok]]
    return out


def norm_joints(K):
    """(T,17,2) pixels (NaN missing) -> BST normalize_joints (box-centre aligned, / box diagonal), 0 where missing."""
    out = np.zeros_like(K, dtype=np.float32)
    for t in range(len(K)):
        ok = np.isfinite(K[t, :, 0])
        if ok.sum() < 2:
            continue
        lo, hi = K[t, ok].min(0), K[t, ok].max(0)
        diag = max(np.linalg.norm(hi - lo), 1e-6)
        out[t, ok] = (K[t, ok] - 0.5 * (lo + hi)) / diag
    return out


def build3d():
    """REP=3d: per hit the same window and players as build(), but each player is the student's 24 SMPL joints plus
    its racket head (pelvis-relative, metres / 1.8, in the hitter's half-court frame), pos is the pelvis on the court
    (/ W, L) and the shuttle is the student's 3D shuttle (/ W, L, 5 m); bones along the SMPL tree and racket-wrist."""
    import hit3d as H3
    import bfmd_stroke as BS
    from birdseye_v2.court import W as CW, L as CL
    sets, _ = H3.S3.load()
    keys = H3.side_keys()
    vocab = json.load(open(f"{H3.S3.tf.BASE}/training_dataset_all_selfdetected.json"))["shot_type_vocab"]
    out, cache, sd, bcache = {}, {}, BS.SRC["sd"], {}
    scale = np.array([CW, CL, 5.0])
    for name in ("train", "sel", "test"):
        J, B, P, S, L, Y, U = [], [], [], [], [], [], []
        for (tag, side), (feat, _, shot) in zip(keys[name], sets[name]):
            m, rt = tag.split("::")
            n = len(feat)
            if m not in cache:
                cache = {m: json.load(open(os.path.join(sd, m, "joints_selfdetected.json")))["rallies"]}
            o = cache[m][rt]
            if BODY_SD:  # bodies from another export (e.g. VIMO) with this export's shuttle and racket
                if m not in bcache:
                    bcache.clear(); bcache[m] = json.load(open(os.path.join(BODY_SD, m, "joints_selfdetected.json")))["rallies"]
                o = dict(o, joints=bcache[m].get(rt, {}).get("joints", {}))
            fit = lambda a, k: np.concatenate([a, np.full((max(0, n - len(a)),) + a.shape[1:], np.nan)])[:n]                 if len(a) else np.full((n,) + k, np.nan)
            sh = fit(np.array(o.get("shuttle", []), float).reshape(-1, 3), (3,))
            body = {}
            for s_ in ("top", "bottom"):
                Jw = fit(np.array(o["joints"].get(s_) or [], float).reshape(-1, 24, 3), (24, 3))
                rk = fit(np.array(o.get("racket_" + s_, []), float).reshape(-1, 3), (3,))
                body[s_] = np.concatenate([Jw, rk[:, None]], 1)  # (n, 25, 3) world
            flip = side == "top"  # hitter's half-court frame: the hitter always on the near (y < L/2) half

            def canon_pts(X):  # world -> canonical world
                X = X.copy()
                if flip:
                    X[..., 0] = CW - X[..., 0]; X[..., 1] = CL - X[..., 1]
                return X
            other = "bottom" if side == "top" else "top"
            for h in np.flatnonzero(shot >= 0):
                a, b = max(0, h - W), min(n, h + W + 1)
                T = 2 * W + 1
                j = np.zeros((T, 2, 25, 3), np.float32); bo = np.zeros((T, 2, len(BONES3D), 3), np.float32)
                p = np.zeros((T, 2, 2), np.float32); s3 = np.zeros((T, 3), np.float32)
                for i, who in enumerate((side, other)):
                    X = canon_pts(body[who][a:b])
                    pel = X[:, 0]
                    rel = (X - pel[:, None]) / 1.8
                    bn = np.stack([rel[:, c] - rel[:, q] for q, c in BONES3D], 1)
                    j[:b - a, i] = np.nan_to_num(rel); bo[:b - a, i] = np.nan_to_num(bn)
                    p[:b - a, i] = np.nan_to_num(np.clip(pel[:, :2] / scale[:2], -0.5, 1.5))
                s3[:b - a] = np.nan_to_num(np.clip(canon_pts(sh[a:b]) / scale, -1, 2))
                J.append(j); B.append(bo); P.append(p); S.append(s3); L.append(b - a); Y.append(shot[h])
                U.append(f"{tag}|{side}|{h}")
        out[name] = dict(joints=np.stack(J), bones=np.stack(B), pos=np.stack(P), shuttle=np.stack(S),
                         video_len=np.array(L), labels=np.array(Y, np.int64), ids=np.array(U))
        print(f"{name}: {len(Y)} clips", flush=True)
    os.makedirs(os.path.dirname(CLIPS), exist_ok=True)
    np.savez_compressed(CLIPS, n_cls=len(vocab), vocab=np.array(vocab), **{f"{k}_{f}": v for k, dd in out.items() for f, v in dd.items()})
    print("wrote", CLIPS, flush=True)


def merge():
    """REP=2d3d: both clip sets side by side (same hits, same order): per player the 2D keypoints (third channel 0)
    and the 3D joints as one point set, both bone sets, and the 2D + 3D shuttle as five channels; pos from 2D."""
    z2 = np.load(CLIPS.replace(REP, "")); z3 = np.load(CLIPS.replace(REP, REP[2:]))  # 2d3d<x>: 2D + clips 3d<x>
    out = {"n_cls": z2["n_cls"]}
    pad = lambda a: np.concatenate([a, np.zeros(a.shape[:-1] + (1,), a.dtype)], -1)
    for n in ("train", "sel", "test"):
        assert (z2[n + "_ids"] == z3[n + "_ids"]).all()
        j2 = z2[n + "_joints"]
        out[n + "_joints"] = np.concatenate([pad(j2), z3[n + "_joints"]], -2)
        out[n + "_bones"] = np.concatenate([pad(_bones(j2)), z3[n + "_bones"]], -2)
        out[n + "_shuttle"] = np.concatenate([z2[n + "_shuttle"], z3[n + "_shuttle"]], -1)
        for k in ("pos", "video_len", "labels", "ids"):
            out[n + "_" + k] = z2[n + "_" + k]
    np.savez_compressed(CLIPS, **out)
    print("wrote", CLIPS, {k: v.shape for k, v in out.items() if k.startswith("test")}, flush=True)


def build():
    if REP == "3d":
        return build3d()
    if REP.startswith("2d3d"):
        return merge()
    import hit3d as H3
    import bfmd_stroke as BS
    from bfmd_export_2d import side_kp
    from birdseye_v2.players import load_pose
    from birdseye_v2.court import W as CW, L as CL
    if SET == "fbs":  # FineBadminton-20K: inputs under outputs/fb20k, clips data/fb20k/clips/<match>_<KKK>.mp4
        BS.SRC.update(out=os.path.join(ROOT, "outputs", "fb20k"),
                      uid=lambda m, rt: f"{m}_{int(rt.split('_r')[1]):03d}")
        import cv2

        def vsize(uid):
            c = cv2.VideoCapture(os.path.join(ROOT, "data", "fb20k", "clips", uid + ".mp4"))
            return (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) if c.isOpened() else None
        BS.video_size = vsize
    sets, _ = H3.S3.load()
    keys = H3.side_keys()
    d = json.load(open(f"{H3.S3.tf.BASE}/training_dataset_all_selfdetected.json"))
    vocab = d["shot_type_vocab"]
    out = {}
    cache = {}
    for name in ("train", "sel", "test"):
        J, P, S, L, Y, U = [], [], [], [], [], []
        for (tag, side), (feat, _, shot) in zip(keys[name], sets[name]):
            m, rt = tag.split("::")
            uid = BS.SRC["uid"](m, rt)
            n = len(feat)
            if uid not in cache:
                vp = os.path.join(BS.SRC["out"], "vimo_bc", uid + ".npz")
                pp = os.path.join(BS.SRC["out"], "pose1280", uid + ".npz")
                cp = os.path.join(BS.SRC["out"], "calib_bc", uid + ".json")
                kp, foot = {}, {}
                size = BS.video_size(uid) or (1280, 720)
                c = json.load(open(cp)).get("cam") if os.path.exists(cp) else None
                Hh = c["size"][1] if c else 720
                cams = BS.frame_cams(uid, n)
                if os.path.exists(vp) and os.path.exists(pp):
                    dv, pose = np.load(vp), load_pose(pp)
                    for s in ("top", "bottom"):
                        if s + "_ankle_px" not in dv.files:
                            continue
                        K = side_kp(pose, dv["frames"], dv[s + "_ankle_px"], Hh)  # (u/H, 0, -v/H)
                        K = at_30fps(K, dv["frames"], n)
                        px = np.stack([K[..., 0] * Hh, -K[..., 2] * Hh], -1)  # (n,17,2) pixels
                        kp[s] = px
                        ank = 0.5 * (px[:, 15] + px[:, 16])
                        g = np.zeros((n, 2), np.float32)
                        for t in range(n):
                            cam = cams[t] if isinstance(cams, list) else cams
                            if cam is not None and np.isfinite(ank[t]).all():
                                q = cam.pixel_to_ground(ank[t][None], z=0.0)[0]
                                g[t] = np.clip([q[0] / CW, q[1] / CL], -0.5, 1.5)
                        foot[s] = g
                sh = np.zeros((n, 2), np.float32)
                rp = os.path.join(BS.SRC["out"], "refined", uid + ".json")
                if os.path.exists(rp):
                    tr = json.load(open(rp))
                    f = np.array(tr["frames"], int)
                    px = np.array(tr["px"], float).reshape(-1, 2)
                    k = (f >= 0) & (f < n) & np.isfinite(px).all(1)
                    sh[f[k]] = px[k] / np.array(size, float)
                cache = {uid: (kp, foot, sh)}  # one rally at a time (sides follow each other)
            kp, foot, sh = cache[uid]
            other = "bottom" if side == "top" else "top"
            for h in np.flatnonzero(shot >= 0):
                a, b = max(0, h - W), min(n, h + W + 1)
                j = np.zeros((2 * W + 1, 2, 17, 2), np.float32)
                p = np.zeros((2 * W + 1, 2, 2), np.float32)
                s_ = np.zeros((2 * W + 1, 2), np.float32)
                for i, who in enumerate((side, other)):
                    if who in kp:
                        j[:b - a, i] = norm_joints(kp[who][a:b])
                        p[:b - a, i] = foot[who][a:b]
                s_[:b - a] = sh[a:b]
                J.append(j); P.append(p); S.append(s_); L.append(b - a); Y.append(shot[h]); U.append(f"{tag}|{side}|{h}")
        out[name] = dict(joints=np.stack(J), pos=np.stack(P), shuttle=np.stack(S), video_len=np.array(L),
                         labels=np.array(Y, np.int64), ids=np.array(U))
        print(f"{name}: {len(Y)} clips", flush=True)
    os.makedirs(os.path.dirname(CLIPS), exist_ok=True)
    np.savez_compressed(CLIPS, n_cls=len(vocab), vocab=np.array(vocab), **{f"{k}_{f}": v for k, dd in out.items() for f, v in dd.items()})
    print("wrote", CLIPS, flush=True)


def _bones(j):
    from preparing_data.shuttleset_dataset import get_bone_pairs
    out = []
    for s, e in get_bone_pairs():
        a, b = j[..., s, :], j[..., e, :]
        out.append(np.where((a != 0) & (b != 0), b - a, 0.0))
    return np.stack(out, -2).astype(np.float32)


def train(model_name, seed):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import get_cosine_schedule_with_warmup
    from preparing_data.shuttleset_dataset import RandomTranslation_batch, get_bone_pairs
    from stroke3d import metrics
    torch.manual_seed(seed); np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device("cuda")
    z = np.load(CLIPS)
    n_cls = int(z["n_cls"])
    xz = np.load(XTEST) if XTEST else None  # cross-broadcast: test on another dataset's clips of the same input
    if xz is not None:  # both label sets -> the shared classes (COMMON); clips of unshared classes dropped
        z = {k: z[k] for k in z.files}
        xz = {k: xz[k] for k in xz.files}
        for d in (z, xz):
            lut = np.array([COMMON.get(str(v), -1) for v in d["vocab"]])
            for n in ("train", "sel", "test"):
                y = lut[d[n + "_labels"]]
                k = y >= 0
                for f in [f for f in list(d) if f.startswith(n + "_")]:
                    d[f] = d[f][k]
                d[n + "_labels"] = y[k]
        n_cls = len(set(COMMON.values()))
    T = 2 * W + 1
    style = "J_only" if model_name == "stgcn" else "JnB_bone"
    has_b = "train_bones" in z  # REP=3d clips carry their own (SMPL-tree) bones
    nb = z["train_bones"].shape[-2] if has_b else len(get_bone_pairs())
    nj, dim, sdim = z["train_joints"].shape[-2], z["train_joints"].shape[-1], z["train_shuttle"].shape[-1]

    def tensors(name, z=z):
        j = z[name + "_joints"]
        if style == "J_only":
            hp = j
        else:
            hp = np.concatenate([j, z[name + "_bones"] if has_b else _bones(j)], -2)
        return [torch.from_numpy(a).to(dev) for a in (hp.astype(np.float32), z[name + "_pos"], z[name + "_shuttle"],
                                                      z[name + "_video_len"], z[name + "_labels"])]
    tr, se, te = tensors("train"), tensors("sel"), tensors("test")
    tx = tensors("test", xz) if xz is not None else None
    frac = float(os.environ.get("TRAIN_FRAC", "1"))
    if frac < 1:  # a fixed random subset of the training MATCHES (same for every representation)
        ms = np.array([u.split("::")[0] for u in z["train_ids"]])
        um = sorted(set(ms)); keep_m = set(np.random.RandomState(0).permutation(um)[:max(1, round(frac * len(um)))])
        k = torch.from_numpy(np.flatnonzero([m in keep_m for m in ms])).to(dev)
        tr = [t[k] for t in tr]
        print(f"TRAIN_FRAC {frac}: {len(keep_m)}/{len(um)} matches, {len(k)} clips", flush=True)
    n_epochs = int(os.environ.get("EPOCHS", "1600")); patience = int(os.environ.get("PATIENCE", "300"))
    bs, lr = 128, (1e-3 if model_name == "stgcn" else 5e-4)
    if model_name in ("bst", "bst_cg_ap"):
        from model.bst import BST, BST_CG_AP
        net = (BST if model_name == "bst" else BST_CG_AP)(in_dim=(nj + nb) * dim, n_class=n_cls, seq_len=T,
                                                         depth_tem=2, depth_inter=1, **HP)
        fwd = lambda hp, pos, sh, vl: net(hp.flatten(-2), sh, pos, vl)
    elif model_name == "tempose":
        from model.tempose import TemPose_TF
        net = TemPose_TF(in_dim=(nj + nb) * dim, n_class=n_cls, seq_len=T, depth_inter=2)
        fwd = lambda hp, pos, sh, vl: net(hp.flatten(-2), pos, sh, vl)
    else:
        from model.stgcn import ST_GCN_18
        net = ST_GCN_18(in_channels=2, num_class=n_cls, graph_cfg={"layout": "coco", "strategy": "spatial"},
                        edge_importance_weighting=True, data_bn=False, tem_kernel_size=9, dropout=0.5)
        fwd = lambda hp, pos, sh, vl: net(hp.permute(0, 4, 1, 3, 2).contiguous())
    if sdim != 2 and hasattr(net, "tcn_shuttle"):  # 3D shuttle: the same TCN on three channels
        from model.tempose import TCN
        c = net.tcn_shuttle
        net.tcn_shuttle = TCN(sdim, [net.d_model // 2, net.d_model], 5, 0.3)
    net = net.to(dev)
    shift = RandomTranslation_batch()
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    steps = (len(tr[-1]) + bs - 1) // bs
    sched = get_cosine_schedule_with_warmup(opt, 400, n_epochs * steps, num_cycles=0.25)

    def predict(ts):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(ts[-1]), 512):
                out.append(fwd(*(t[i:i + 512] for t in ts[:4])).argmax(1).cpu().numpy())
        return np.concatenate(out)
    best, best_ep, best_state, wait = -1, -1, None, 0
    t0 = time.time()
    for ep in range(n_epochs):
        net.train()
        perm = torch.randperm(len(tr[-1]), device=dev)
        for b in range(steps):
            i = perm[b * bs:(b + 1) * bs]
            hp, pos, sh, vl, y = (t[i] for t in tr)
            if style == "J_only":
                hp = shift(hp)
            else:
                hp = torch.cat([shift(hp[..., :-nb, :].contiguous()), hp[..., -nb:, :]], -2)
            loss = F.cross_entropy(fwd(hp, pos, sh, vl), y, label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        acc, f1 = metrics(predict(se), se[-1].cpu().numpy())
        wait += 1
        if f1 > best:
            best, best_ep, best_state, wait = f1, ep, copy.deepcopy(net.state_dict()), 0
        if ep % 50 == 0:
            print(f"{model_name} s{seed} epoch {ep}: sel acc={acc:.3f} macroF1={f1:.3f} (best {best:.3f} @ {best_ep}) "
                  f"{time.time() - t0:.0f}s", flush=True)
        if wait >= patience:
            break
    net.load_state_dict(best_state)
    acc, f1 = metrics(predict(te), te[-1].cpu().numpy())
    r = {"seed": seed, "epoch": best_ep, "sel_f1": best, "test_acc": acc, "test_macro_f1": f1}
    if tx is not None:
        r["x_acc"], r["x_macro_f1"] = metrics(predict(tx), tx[-1].cpu().numpy())
        print(f"XTEST {os.path.basename(XTEST)}: acc={r['x_acc']:.3f} macroF1={r['x_macro_f1']:.3f} "
              f"(n={len(tx[-1])}) | in-domain acc={acc:.3f} macroF1={f1:.3f}", flush=True)
    return r


def main():
    if os.environ.get("STAGE", "train") == "build":
        return build()
    model_name = os.environ.get("MODEL", "bst")
    tag = os.environ.get("TAG", f"{model_name}{'' if REP == '2d' else REP}_W{W}")
    runs = []
    for seed in [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]:
        r = train(model_name, seed)
        runs.append(r)
        print(f"RESULT {tag} seed {seed}: epoch {r['epoch']} sel macroF1={r['sel_f1']:.3f} | test at true hits "
              f"acc={r['test_acc']:.3f} macroF1={r['test_macro_f1']:.3f}", flush=True)
    if XTEST:
        XA = np.array([r["x_acc"] for r in runs]); XF = np.array([r["x_macro_f1"] for r in runs])
        print(f"XMEAN {tag}: acc {100 * XA.mean():.1f} +- {100 * XA.std(ddof=1):.1f}, macroF1 {100 * XF.mean():.1f} "
              f"+- {100 * XF.std(ddof=1):.1f}", flush=True)
    A = np.array([r["test_acc"] for r in runs]); Fm = np.array([r["test_macro_f1"] for r in runs])
    sd = lambda a: 100 * a.std(ddof=1) if len(a) > 1 else 0
    print(f"MEAN {tag}: acc {100 * A.mean():.1f} +- {sd(A):.1f}, macroF1 {100 * Fm.mean():.1f} +- {sd(Fm):.1f}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({"tag": tag, "model": model_name, "W": W, "runs": runs}, open(os.path.join(OUT, tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()
