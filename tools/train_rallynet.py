"""Train RallyNet (birdseye_v2/rallynet.py) on broadcast rallies without 3D labels; evaluate on DroneData against
two-view ground truth with the demo's metrics (tools/export_viewer.py): player position vs the top-view
annotations, shuttle 3D error, labelled hits recovered within 3 frames.

Train sets: outputs/rally_obs/{bfmd,fb,fb20k}; DroneData (outputs/rally_obs/drone) is never trained on.
usage: python tools/train_rallynet.py [train|eval]   (env STEPS=20000, CROP=160, BATCH=12, TAG=v0, SETS=bfmd+fb+fb20k)"""
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
ARCH = os.environ.get("ARCH", "rallynet")  # rallynet | rallyphys (birdseye_v2/rallyphys.py)


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


VTEACH = float(os.environ.get("VTEACH", "0"))  # weight of the VIMO bone-direction teacher (training only)
SHUTTLE3D = float(os.environ.get("SHUTTLE3D", "0"))  # weight of the two-view 3D shuttle on DroneData training matches
CONTACT3D = float(os.environ.get("CONTACT3D", "0"))  # weight of the physical contact points (racket head at labelled hits)
RACKET_TOP = float(os.environ.get("RACKET_TOP", "0"))  # weight of the top drone's racket rays (DroneData training matches)
TEACH_DIR = os.environ.get("TEACH_DIR", "teach")  # outputs/cctv/<dir>: teacher labels (tools/cctv_teach.py, tools/teach_export.py)
TEACH = float(os.environ.get("TEACH", "0"))  # two-view physics teacher (tools/cctv_teach.py): dense 3D + its hits on CCTV rallies
STUDENT = os.environ.get("STUDENT", "0") == "1"  # evaluate the network's own shuttle (no solver, no rules)
VCAM = float(os.environ.get("VCAM", "0"))  # share of CCTV training rallies seen from a virtual camera (birdseye_v2/vcam.py)
XVIEW = float(os.environ.get("XVIEW", "0"))  # weight of the CCTV side camera's shuttle detections on the model's 3D shuttle
CCTV3D = float(os.environ.get("CCTV3D", "0"))  # weight of the CCTV two-view shuttle at the network's own hits (tools/cctv_tri.py)
RT = 8  # top-view racket rays kept per frame


def top_rackets(uid, T):
    """The top drone's racket boxes (tools/top_racket.py) as rays in the court frame at the back view's frames:
    (T,3) camera centre, (T,RT,3) unit rays through the box centres, (T,RT) mask."""
    from birdseye_v2.court import Camera
    C, D, M = np.zeros((T, 3), np.float32), np.zeros((T, RT, 3), np.float32), np.zeros((T, RT), np.float32)
    m_, r_ = uid.split("__")
    rp, cp = os.path.join(ROOT, "outputs", "racket_top", m_, r_ + ".npz"), os.path.join(ROOT, "outputs", "calib_bc", m_, r_ + ".json")
    g = json.load(open(os.path.join(ROOT, "outputs", "gt", m_, r_ + ".json")))
    if not (os.path.exists(rp) and os.path.exists(cp) and g.get("sync")):
        return C, D, M
    cal = json.load(open(cp))
    if not cal.get("top"):
        return C, D, M
    top = Camera.from_dict(cal["top"])
    d = np.load(rp)
    B = dict(zip(d["frames"].tolist(), np.split(d["boxes"], np.cumsum(d["counts"])[:-1])))
    off = g["sync"]["offset"]
    for fb in range(T):
        bx = B.get(int(round(fb - off)))
        if bx is None:
            continue
        bx = bx[bx[:, 4] > 0.3][:RT]
        if not len(bx):
            continue
        c, dr = top.rays(np.c_[(bx[:, 0] + bx[:, 2]) / 2, (bx[:, 1] + bx[:, 3]) / 2])
        C[fb], D[fb, :len(bx)], M[fb, :len(bx)] = np.ravel(c), dr, 1.0
    return C, D, M


def contact_points(G, hits):
    """Physical contact point of each labelled hit from the two-view shuttle: one gravity + drag flight fitted to the
    first GT samples after the hit (>= 5, clear of the contact by a frame), integrated from the hit -> {hit: xyz}."""
    from scipy.optimize import least_squares
    from birdseye_v2.shuttle3d import simulate, FPS
    out = {}
    H = sorted(hits)
    for i, h in enumerate(H):
        nxt = H[i + 1] if i + 1 < len(H) else 1e9
        R = G[(G[:, 0] > h + 1) & (G[:, 0] < nxt - 1)][:12]
        if len(R) < 5:
            continue
        t = (R[:, 0] - h) / FPS
        v0 = (R[3, 1:] - R[0, 1:]) / ((R[3, 0] - R[0, 0]) / FPS)
        q = least_squares(lambda q: (simulate(q[:3], q[3:], t) - R[:, 1:]).ravel(), np.r_[R[0, 1:] - v0 * t[0], v0]).x
        if np.sqrt(np.mean((simulate(q[:3], q[3:], t) - R[:, 1:]) ** 2)) < 0.1:
            out[h] = q[:3]
    return out
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
    if st not in ("bfmd", "fb", "fb20k", "cctv") or not os.path.exists(vp):
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


def side_rays(uid, T):
    """The rally's session side camera (frame and frame shift as chosen by tools/cctv_tri.py) and its rays through the
    side view's refined shuttle track, indexed by TV frame (side frame = TV frame + shift)."""
    import cctv_tri as CT
    C, D, V = np.zeros((T, 3), np.float32), np.zeros((T, 3), np.float32), np.zeros(T, np.float32)
    trk = CT.track(os.path.join(ROOT, "outputs", "cctv_side", "refined", uid + ".json"))
    sc = CT.side_camera(uid)
    if not trk or sc is None:
        return C, D, V
    cam, sh = sc
    ks = [k for k in range(T) if k + sh in trk]
    if ks:
        c, d = cam.rays(np.array([trk[k + sh] for k in ks]))
        C[:] = c
        D[ks], V[ks] = d, 1.0
    return C, D, V


def load_feats(path, virtual=False):
    """Model inputs + hit labels (T,P) where the rally's set has them (slot side from its median foot y).
    virtual (training only): with probability VCAM a CCTV rally's inputs are re-made from a virtual camera (birdseye_v2/vcam.py)."""
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
    f["hit_any_m"] = np.ones(T, np.float32)  # frames where the hit/no-hit label holds (the teacher masks its unsure hits)
    if st == "drone" and uid.startswith(TRAIN_DRONE):  # DroneData training matches: hit frames, hitter not given
        for l in json.load(open(path[:-4] + "_gt.json"))["hits"]:
            if 0 <= int(round(l)) < T:
                f["hit_any"][int(round(l))] = 1.0
        f["hit_any_has"] = np.float32(1)
    f["J3d"] = np.zeros((T, RN.P, RN.J, 3), np.float32)
    f["J3d_mask"] = np.zeros((T, RN.P, RN.J), np.float32)
    if st == "drone" and uid.startswith(TRAIN_DRONE):  # two-drone triangulated joints, matched to slots by position
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
    f["S3d"] = np.zeros((T, 3), np.float32)
    f["S3d_mask"] = np.zeros(T, np.float32)
    if SHUTTLE3D > 0 and st == "drone" and uid.startswith(TRAIN_DRONE):  # two-view shuttle at the whole frames
        G = np.array(json.load(open(path[:-4] + "_gt.json"))["shuttle"], float)
        if len(G) > 3:
            t = G[:, 0]
            k = np.arange(T, dtype=float)
            i = np.clip(np.searchsorted(t, k), 1, len(t) - 1)
            ok = (k >= t[0]) & (k <= t[-1]) & (t[i] - t[i - 1] <= 1.5)  # inside a run of GT samples (no long gaps)
            for c in range(3):
                f["S3d"][:, c] = np.interp(k, t, G[:, 1 + c])
            f["S3d_mask"] = ok.astype(np.float32)
    f["Con3d"] = np.zeros((T, 3), np.float32)
    f["Con3d_mask"] = np.zeros(T, np.float32)
    if CONTACT3D > 0 and st == "drone" and uid.startswith(TRAIN_DRONE):  # physical contacts at the labelled hits
        gtj = json.load(open(path[:-4] + "_gt.json"))
        G = np.array(gtj["shuttle"], float).reshape(-1, 4)
        for h, X in contact_points(G, gtj["hits"]).items():
            k = int(round(h))
            if 0 <= k < T:
                f["Con3d"][k], f["Con3d_mask"][k] = X, 1.0
    if CCTV3D > 0:  # CCTV rallies: TV + fixed side camera triangulated shuttle, at the frames both views saw it
        f["S3c"] = np.zeros((T, 3), np.float32)
        f["S3c_mask"] = np.zeros(T, np.float32)
        tp = os.path.join(ROOT, "outputs", "cctv", "tri", uid + ".json")
        if st == "cctv" and os.path.exists(tp):
            for k, x, y, z in json.load(open(tp))["shuttle"]:
                if 0 <= k < T:
                    f["S3c"][k], f["S3c_mask"][k] = (x, y, z), 1.0
    if TEACH > 0:  # CCTV training rallies: the teacher's chain of flights at every frame, and its hits (when, not who)
        f["Tch"], f["Tch_mask"] = np.zeros((T, 3), np.float32), np.zeros(T, np.float32)
        tp = os.path.join(ROOT, "outputs", "cctv", TEACH_DIR, uid + ".json")
        if st == "cctv" and os.path.exists(tp):
            tj = json.load(open(tp))
            for a_, b_ in tj.get("hit_mask", []):  # tools/teach_export.py: decoded hits the teacher is unsure of
                f["hit_any_m"][max(int(a_), 0):max(int(b_) + 1, 0)] = 0.0
            for k, x, y, z in tj["S"]:
                if 0 <= k < T:
                    f["Tch"][k], f["Tch_mask"][k] = (x, y, z), 1.0
            for h in tj["hits"]:
                if 0 <= int(round(h)) < T:
                    f["hit_any"][int(round(h))] = 1.0
            f["hit_any_has"] = np.float32(1)
            if os.environ.get("TCH_SRC") == "tri":  # label ablation: plain TV+side triangulation instead of the teacher's shuttle
                f["Tch"], f["Tch_mask"] = np.zeros((T, 3), np.float32), np.zeros(T, np.float32)
                trp = os.path.join(ROOT, "outputs", "cctv", "tri", uid + ".json")
                if os.path.exists(trp):
                    for k, x, y, z in json.load(open(trp))["shuttle"]:
                        k = int(round(k))
                        if 0 <= k < T:
                            f["Tch"][k], f["Tch_mask"][k] = (x, y, z), 1.0
    if float(os.environ.get("BODY24", "0")) > 0:  # CCTV training rallies: the teacher's multi-camera TRAM bodies (tools/body_mv.py)
        f["B24"] = np.zeros((T, RN.P, 24, 3), np.float32)
        f["B24_mask"] = np.zeros((T, RN.P), np.float32)
        bp = os.path.join(ROOT, "outputs", "cctv", os.environ.get("BODY_DIR", "body_mvt2"), uid + ".npz")
        if st == "cctv" and os.path.exists(bp):
            vb = np.load(bp)
            for s in ("bottom0", "bottom1", "top0", "top1"):
                if s not in vb.files:
                    continue
                ok_cam = (vb[s + "_nobs"] > 0).sum(1) >= 2 if s + "_nobs" in vb.files else np.ones(len(vb["frames"]), bool)
                for i, fr in enumerate(vb["frames"]):
                    Jt = vb[s][i]
                    if not (0 <= fr < T and ok_cam[i] and np.isfinite(Jt).all()):
                        continue
                    dist = np.where(f["present"][fr] > 0, np.linalg.norm(f["root"][fr] - Jt[0, :2], axis=1), 1e9)
                    q = int(np.argmin(dist))
                    if dist[q] < 0.6:
                        f["B24"][fr, q], f["B24_mask"][fr, q] = Jt, 1.0
    if XVIEW > 0:  # CCTV rallies: rays of the fixed side camera through its shuttle detections, at the TV frame
        f["Xs_C"], f["Xs_dir"], f["Xs_vis"] = (side_rays(uid, T) if st == "cctv" else
                                               (np.zeros((T, 3), np.float32), np.zeros((T, 3), np.float32), np.zeros(T, np.float32)))
    if RACKET_TOP > 0:
        f["Tk_C"], f["Tk_dir"], f["Tk_m"] = (top_rackets(uid, T) if st == "drone" and uid.startswith(TRAIN_DRONE) else
                                             (np.zeros((T, 3), np.float32), np.zeros((T, RT, 3), np.float32), np.zeros((T, RT), np.float32)))
    if virtual and VCAM > 0 and st == "cctv" and "B24" in f and "Tch" in f and np.random.default_rng().random() < VCAM:
        from birdseye_v2 import vcam
        r = vcam.synth(f["B24"], f["B24_mask"], f["Tch"], f["Tch_mask"], 1.0 / float(f["dt"]), np.random.default_rng())
        if r is not None:  # the same slots, labels unchanged: only what the camera saw is new
            f.update(RN.featurize(r[0]))
            f["vcam"] = np.float32(1)
    f.setdefault("vcam", np.float32(0))
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
            f = load_feats(self.paths[rng.integers(len(self.paths))], virtual=True)
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
    sp = os.path.join(ROOT, "outputs", "cctv", "split.json")
    if os.path.exists(sp):  # nor the held-out CCTV matches (tools/cctv_split.py)
        test |= set(json.load(open(sp))["test"])
    if os.environ.get("HOLDOUT"):  # a file of rally uids kept out of training too (e.g. the viewer's rallies)
        test |= set(open(os.environ["HOLDOUT"]).read().split())
    paths = [p for s in SETS for p in glob.glob(os.path.join(OBS, s, "*.npz")) if os.path.basename(p)[:-4] not in test]
    if os.environ.get("CCTV_FRAC"):  # learning curve: keep a fixed random fraction of the CCTV training rallies
        cc = sorted(p for p in paths if os.sep + "cctv" + os.sep in p)
        keep = set(np.random.RandomState(0).permutation(cc)[:int(round(float(os.environ["CCTV_FRAC"]) * len(cc)))])
        paths = [p for p in paths if p not in set(cc) or p in keep]
        print("CCTV_FRAC", os.environ["CCTV_FRAC"], ":", len(keep), "of", len(cc), "CCTV rallies", flush=True)
    if os.environ.get("DRONE_DEV", "1") == "1":  # DroneData training matches (dev BD12-14 + DRONE_TRAIN) with hit labels, x3
        paths += [p for p in glob.glob(os.path.join(OBS, "drone", "*.npz")) if os.path.basename(p).startswith(TRAIN_DRONE)] * 3
    print(len(paths), "training rallies from", SETS, flush=True)
    model = new_model()
    if os.environ.get("INIT"):  # fine-tune from a trained checkpoint (outputs/rallynet/<INIT>.pt)
        sd = torch.load(os.path.join(OUT, os.environ["INIT"] + ".pt"), map_location=DEV)
        print("INIT from", os.environ["INIT"], model.load_state_dict(sd, strict=False), flush=True)
    print(ARCH, sum(p.numel() for p in model.parameters()) / 1e6, "M params", flush=True)
    LR = float(os.environ.get("LR", "3e-4"))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
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
            has = has & ((b["hit_any_m"] + 1 - b["cam_ok"]).min(-1).values > 0)  # not where the teacher left a hit unlabelled
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
        if step % int(os.environ.get("SAVE_EVERY", "2000")) == 0 or step == STEPS - 1:
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
    out = {k: v / wsum.reshape(-1, *[1] * (v.ndim - 1)) for k, v in acc.items()}
    if HITNET:  # who hits when: the hit head averaged with the dedicated hit model (birdseye_v2/hitnet.py)
        global _HN
        if _HN is None:
            import train_hitnet as TH
            _HN = (TH, TH.load(HITNET))
        out["hit"] = (out["hit"] + HITNET_W * _HN[0].infer(_HN[1], f)) / (1 + HITNET_W)
    return out


HITNET = os.environ.get("HITNET", "")  # tag of a trained HitNet (outputs/hitnet/<tag>.pt) to take the hits from
HITNET_W = float(os.environ.get("HITNET_W", "0.5"))  # its weight against the hit head's logits
_HN = None


def hits_from(prob, thr=None, nms=6):
    thr = HIT_THR if thr is None else thr
    from scipy.signal import find_peaks
    pk, _ = find_peaks(prob, height=thr, distance=nms)
    return pk


DEV_MATCHES = ("BD12_", "BD13_", "BD14_")  # DroneData dev matches (thresholds are chosen here); never in the scores
# more DroneData matches moved from the scores into training (their hits and two-drone poses), e.g. DRONE_TRAIN=BD26+BD27
TRAIN_DRONE = DEV_MATCHES + tuple(m + "_" for m in os.environ.get("DRONE_TRAIN", "").split("+") if m)
HIT_THR = float(os.environ.get("HIT_THR", "0.5"))
DEMO = ["BD15__Rally-39", "BD15__Rally-31", "BD17__Rally-1", "BD19__Rally-32", "BD21__Rally-9", "BD23__Rally-27"]


def eval_paths(full=False):
    """All DroneData rallies, or a fixed 60 (the six demo rallies + 54 others) for checks during training.
    EVAL_SET=cctv: the held-out CCTV matches with two-view ground truth (tools/cctv_split.py)."""
    if os.environ.get("EVAL_SET") == "cctv":
        return sorted(p for p in glob.glob(os.path.join(OBS, "cctv", "*.npz")) if os.path.exists(p[:-4] + "_gt.json"))
    ps = [p for p in sorted(glob.glob(os.path.join(OBS, "drone", "*.npz"))) if not os.path.basename(p).startswith(EVAL_SKIP)]
    if full:
        return ps
    demo = [p for p in ps if os.path.basename(p)[:-4] in DEMO]
    rest = [p for p in ps if p not in demo]
    return demo + [rest[i] for i in np.random.default_rng(0).choice(len(rest), min(54, len(rest)), replace=False)]


PHYS = os.environ.get("PHYS", "1") == "1" and ARCH == "rallynet"  # RallyPhys integrates its own shuttle
CHAIN_SHOTS = os.environ.get("CHAIN_SHOTS", "0") == "1"  # chain contacts at the per-shot fit's shots (else the hits)
REFINE = os.environ.get("REFINE", "1") == "1"  # add the network's sub-threshold hits where one flight cannot explain a shot
REFINE_PX = 20.0  # ... only when some shot's flight misses its detections by more than this (median px)
SPLIT_CHECK = os.environ.get("SPLIT_CHECK", "0") == "1"  # drop the track-driven splits when they make the fit unphysical
OUTSIDE = os.environ.get("OUTSIDE", "0") == "1"  # ... or flies out of the hall (beyond the court surrounds / into the floor)
MIN_SHOT = 5  # frames: an added hit is at least this far from the shot's ends
SERVE_CLAMP = os.environ.get("SERVE_CLAMP", "1") == "1"  # hits before the track starts move to its start (no gap to fill)
SIDE_VOTE = os.environ.get("SIDE_VOTE", "1") == "1"  # the network's hitters vote the serve side (shuttle3d.fit_rally)
LAND_CHECK = os.environ.get("LAND_CHECK", "0") == "1"  # refit without the landing when the fit ends high above the floor
STATIC_END = float(os.environ.get("STATIC_END", "0"))  # px: trailing shots this still (p90 spread) lie on the floor (0 = off)
STATIC_TRIM = os.environ.get("STATIC_TRIM", "0") == "1"  # the shuttle never jumps: no floor point far from the flight
FLY_ON = os.environ.get("FLY_ON", "0") == "1"  # after the track the last flight flies on to the floor (no drop)
STATIC_DROP = os.environ.get("STATIC_DROP", "0") == "1"  # a last "hit" followed by a still shuttle is its landing: drop it
REST_FLOOR = os.environ.get("REST_FLOOR", "1") == "1"  # a still shuttle after the last shot lies on the floor
ANCHOR = os.environ.get("ANCHOR", "racket")  # hit anchors: the hitters' racket heads, or "knot" (the network's contact point)
# the far hitter's racket head sits FAR_SHIFT m beyond the contact along the camera ray (BD26-30 drone train: +1.01,
# near hitter -0.01): pull its anchor back toward the camera by that much
FAR_SHIFT = float(os.environ.get("FAR_SHIFT", "0"))


def physics_shuttle(p, f, out, return_shots=False):
    """The physics output layer: one gravity + drag flight per shot (birdseye_v2.pipeline.reconstruct_rally),
    split at the network's hits and started at the network's racket head of the player whose hand is nearest the
    shuttle in the image. -> (T,3) shuttle positions."""
    from birdseye_v2 import pipeline as PL, shuttle3d as S3
    from birdseye_v2.court import Camera, NET_Y, L, W
    uid = os.path.basename(p)[:-4]
    if "__" in uid:  # DroneData: the back view's camera and shuttle track
        cam = Camera.from_dict(json.load(open(p[:-4] + "_gt.json"))["cam"])
        m, r = uid.split("__")
        tr = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))["back_track"]
    else:  # broadcast sets (CCTV test): the clip's own camera and refined track
        base = os.path.join(ROOT, "outputs", os.path.basename(os.path.dirname(p)))
        cam = Camera.from_dict(json.load(open(os.path.join(base, "calib_bc", uid + ".json")))["cam"])
        tr = json.load(open(os.path.join(base, "refined", uid + ".json")))
    fr = np.array(tr["frames"], float)
    px = np.array(tr["px"], float).reshape(-1, 2)
    S3.FPS = float(round(1.0 / float(f["dt"])))  # the footage's frame rate (CCTV 25, DroneData / FBBench 30)
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
        if ANCHOR == "knot" and "knot" in out and np.isfinite(out["knot"][k]).all():  # the network's own contact point
            K = out["knot"][k]
            return [{"hand": K, "side": int(K[1] > NET_Y), "px": 0.0, **({"far_p": far_p} if SIDE_VOTE else {})}]
        for q in np.where(pres[k])[0]:
            h = wrist[k, q] + RN.GRIP_TO_HEAD * u[k, q]
            if not np.isfinite(h).all():
                continue
            if FAR_SHIFT and jt[k, q, [15, 16], 1].mean() > NET_Y:
                r_ = h - cam_.center
                h = h - FAR_SHIFT * r_ / np.linalg.norm(r_)
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

    def inside(rec):  # a fit that leaves the hall has flown off along its rays (or was launched faster than any smash)
        X = rec["fit"].position(np.arange(int(fr[0]), int(fr[-1]) + 1, dtype=float))
        X = X[np.isfinite(X).all(1)]
        v0 = rec["fit"].params[:6 * len(rec["fit"].shots)].reshape(-1, 6)[:, 3:]
        return bool(np.all((X[:, 0] > -3) & (X[:, 0] < W + 3) & (X[:, 1] > -3) & (X[:, 1] < L + 3) & (X[:, 2] < 12))
                    and (not OUTSIDE or np.linalg.norm(v0, axis=1).max() < 110.0))

    def shot_inside(rec, a, b):  # one shot (away from its ends) stays over the court's surrounds and above the floor
        X = rec["fit"].position(np.arange(np.ceil(a) + 3, np.floor(b) - 2, dtype=float))
        X = X[np.isfinite(X).all(1)]
        return bool(np.all((X[:, 0] > -1.0) & (X[:, 0] < W + 1.0) & (X[:, 1] > -0.8) & (X[:, 1] < L + 0.8) & (X[:, 2] > -0.5)))
    PL._candidates = cands
    try:
        def recon(h):  # the shots' fit; a track-driven split (SPLIT_GIVEN) must keep the fit physical, else no split
            r_ = PL.reconstruct_rally(cam, fr, px, [np.zeros((0, 17, 3))] * T, hits=h)
            if PL.SPLIT_GIVEN > 0 and SPLIT_CHECK and not (inside(r_) and all(shot_inside(r_, a, b) for a, b in r_["fit"].shots)):
                sg, PL.SPLIT_GIVEN = PL.SPLIT_GIVEN, 0.0
                try:
                    r_ = PL.reconstruct_rally(cam, fr, px, [np.zeros((0, 17, 3))] * T, hits=h)
                finally:
                    PL.SPLIT_GIVEN = sg
            return r_
        if STATIC_DROP and STATIC_END > 0:  # a last "shot" that hardly moves in the image (the shuttle at rest, or a far
            # shuttle coming down) is no shot: the flight before it continues through its detections and lands
            while len(hits) > 1:
                m_ = fr >= hits[-1]
                P_ = px[m_]
                if m_.sum() < 6 or np.percentile(np.linalg.norm(P_ - np.median(P_, 0), axis=1), 90) > STATIC_END:
                    break
                hits.pop()  # its detections stay: the flight before flies through them and lands
        rec = recon(hits)
        # a shot one flight cannot explain hides a hit the network missed: add one at the shot's strongest
        # sub-threshold hit peak, keep it if the fit explains the track better and stays in the hall
        for _ in range(4 if REFINE else 0):
            w = shot_px(rec)
            ins = inside(rec)
            add = []
            for (a, b), e in zip(rec["fit"].shots, w):
                lo, hi = int(a) + MIN_SHOT, int(b) - MIN_SHOT
                # ... or (OUTSIDE) the shot flies out of the hall: its pixels fit, its depth cannot be right
                bad = e > REFINE_PX or (OUTSIDE and not shot_inside(rec, a, b))
                if bad and hi > lo and hp[lo:hi].max() > 0.05:
                    add.append(float(lo + np.argmax(hp[lo:hi])))
                elif OUTSIDE and hi > lo and not shot_inside(rec, a, b):  # no hit candidate: where two flights fit best
                    sp_ = S3.split_unexplained(cam, rec["frames"], rec["px"], [(a, b)], max_rms=-1.0, max_rounds=1)
                    if len(sp_) == 2:
                        add.append(float(sp_[0][1]))
            if not add:
                break
            try:
                rec2 = recon(sorted(hits + add))
            except ValueError:
                break
            better = np.minimum(shot_px(rec2), 200).sum() < np.minimum(w, 200).sum()
            if not inside(rec2) or not (better or (OUTSIDE and not ins and np.minimum(shot_px(rec2), 200).sum()
                                                   < 1.5 * np.minimum(w, 200).sum() + 10)):
                break
            rec, hits = rec2, sorted(hits + add)
        if LAND_CHECK and rec.get("landing") is not None:  # a "landing" the fit itself puts high in the air is a slow
            b_ = rec["fit"].shots[-1][1]                    # far shot still flying (it barely moves in the image)
            if rec["fit"].position(np.array([float(b_)]))[0][2] > 1.0:
                lf0, lf1 = S3.landing_frame, PL.landing_frame
                S3.landing_frame = PL.landing_frame = lambda *a, **k: None
                try:
                    rec2 = recon(hits)
                finally:
                    S3.landing_frame, PL.landing_frame = lf0, lf1
                if inside(rec2):
                    rec = rec2
        if STATIC_DROP and STATIC_END > 0:  # ... also a still last shot the fit itself made (a split, an added hit)
            for _ in range(3):
                sh_ = rec["fit"].shots
                if len(sh_) < 2:
                    break
                a_, b_ = sh_[-1]
                m_ = (rec["frames"] >= a_) & (rec["frames"] <= b_)
                P_ = rec["px"][m_]
                if m_.sum() < 6 or np.percentile(np.linalg.norm(P_ - np.median(P_, 0), axis=1), 90) > STATIC_END:
                    break
                hits = [h for h in hits if h < a_ - 1]
                try:
                    rec2 = recon(hits)
                except ValueError:
                    break
                if not inside(rec2):
                    break
                rec = rec2
    except ValueError:  # a degenerate fit: keep the network's own shuttle
        return (out["shuttle"], None) if return_shots else out["shuttle"]
    finally:
        PL._candidates = old
    S = rec["fit"].position(np.arange(T, dtype=float), rest_until=T - 1, fly_on=FLY_ON)
    ok = np.isfinite(S).all(1)
    if ok.any():  # outside the fitted shots the shuttle holds its nearest fitted position (no network fallback)
        idx = np.where(ok)[0]
        S = S[idx[np.clip(np.searchsorted(idx, np.arange(T)), 0, len(idx) - 1)]] if not ok.all() else S
        S = np.where(ok[:, None], rec["fit"].position(np.arange(T, dtype=float), rest_until=T - 1, fly_on=FLY_ON), S)
    else:
        S = out["shuttle"]
    if STATIC_END > 0:  # trailing "shots" whose detections hardly move are the shuttle lying on the floor (landed and
        # bounced; a false hit at the landing): a still pixel fits a flight along its own ray, so cut the ray at the floor
        sh, a0 = rec["fit"].shots, None
        for a, b in sh[::-1]:
            m = (rec["frames"] >= a) & (rec["frames"] <= b)
            P = rec["px"][m]
            if m.sum() < 6 or np.percentile(np.linalg.norm(P - np.median(P, 0), axis=1), 90) > STATIC_END:
                break
            a0 = a
        if a0 is not None and a0 > sh[0][0]:
            for k in range(int(np.ceil(a0)), T):
                uv = PL._interp(rec["frames"], rec["px"], float(k))
                c_, d_ = cam.rays(np.asarray(uv, float).reshape(1, 2))
                if d_[0][2] < -1e-3:
                    X = np.ravel(c_) + (0.03 - np.ravel(c_)[2]) / d_[0][2] * d_[0]
                    if -2 < X[0] < W + 2 and -2 < X[1] < L + 2:
                        S[k] = X
    if REST_FLOOR:  # after the last shot, a shuttle that sits still in the image lies on the floor: cut its ray there
        vis = f["sh_vis"] > 0
        end = rec["fit"].shots[-1][1]
        for k in range(int(np.ceil(end)), T):
            if not vis[k] or f["sh_dir"][k][2] > -1e-3:
                continue
            nb = [j for j in range(k - 3, k + 4) if 0 <= j < T and vis[j]]
            still = max(np.degrees(np.arccos(np.clip(f["sh_dir"][j] @ f["sh_dir"][k], -1, 1))) for j in nb) < 0.15
            if still:
                lam = (0.03 - f["C"][k][2]) / f["sh_dir"][k][2]
                X = f["C"][k] + lam * f["sh_dir"][k]
                if -2 < X[0] < W + 2 and -2 < X[1] < L + 2:
                    S[k] = X
    if STATIC_TRIM:
        S = no_teleport(S)
    return (S, [tuple(map(float, s)) for s in rec["fit"].shots]) if return_shots else S


def no_teleport(S, max_step=2.0, n=10):
    """The shuttle never jumps: where it moves further in one frame than any shot carries it (a flight's end and the
    floor point after it disagree), the n frames before the jump bend smoothly onto the position after it."""
    S = S.copy()
    st = np.linalg.norm(np.diff(S, axis=0), axis=1)
    nb = np.maximum(np.r_[0.0, st[:-1]], np.r_[st[1:], 0.0])  # a smash moves this fast frame after frame; a jump once
    for k in np.where(np.isfinite(st) & (st > max_step) & (st > 3 * np.nan_to_num(nb)))[0]:
        a = max(k - n, 0)
        d = S[k + 1] - S[k]
        w = ((np.arange(a, k + 1) - a + 1) / (k - a + 1)) ** 2  # eases in
        S[a:k + 1] = S[a:k + 1] + w[:, None] * d
    return S


CHAIN = os.environ.get("CHAIN", "0") == "1"  # one continuous chain refit after the per-shot fit (60-check: 0.394 -> 0.409, off)


def rallyphys_shuttle(p, f, out):
    """RallyPhys's shuttle: the hitter-anchored joint fit of the shots (optionally then one exact continuous chain)."""
    from birdseye_v2 import rallyphys
    S0, shots = physics_shuttle(p, f, out, return_shots=True)
    if not CHAIN:
        return S0
    hp = 1 - np.prod(1 - 1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    return rallyphys.chain_refit(S0, hits_from(hp, nms=8), f, shots=shots if CHAIN_SHOTS else None)


EVAL_SKIP = TRAIN_DRONE
BODY = list(range(5, 17))  # COCO shoulders..ankles (the face is not scored)
ARMS = [7, 8, 9, 10]


def pose_err(J, f, uid):
    """3D pose against the two-drone triangulated skeletons (outputs/pose3d, scored matches only): each GT skeleton
    goes to the predicted player standing nearest it (< 0.6 m); errors are root-relative (hip centre removed), over
    the body joints and over elbows + wrists. Per-rally medians over joints seen in both views."""
    m_, r_ = uid.split("__") if "__" in uid else ("", uid)
    tp = os.path.join(ROOT, "outputs", "pose3d", m_, r_ + ".npz")
    if not m_ or not os.path.exists(tp):
        return {"pose_rel": None, "arm_rel": None, "pose_n": 0}
    d = np.load(tp)
    T = len(J)
    root = J[:, :, [11, 12]].mean(2)
    eb, ea = [], []
    for fr, G in zip(d["frames"], d["J"]):
        if not 0 <= fr < T or not np.isfinite(G[[11, 12]]).all():
            continue
        g0 = G[[11, 12]].mean(0)
        dist = np.where(f["present"][fr] > 0, np.linalg.norm(root[fr, :, :2] - g0[:2], axis=1), 1e9)
        q = int(np.argmin(dist))
        if dist[q] >= 0.6:
            continue
        e = np.linalg.norm((J[fr, q] - root[fr, q]) - (G - g0), axis=1)
        eb += [x for x in e[BODY] if np.isfinite(x)]
        ea += [x for x in e[ARMS] if np.isfinite(x)]
    r3 = lambda v: round(float(np.median(v)), 3) if v else None
    return {"pose_rel": r3(eb), "arm_rel": r3(ea), "pose_n": len(eb)}


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
    elif STUDENT:  # the network's own shuttle, one forward pass
        S = out["shuttle"]
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
    pose = pose_err(out["joints"], f, os.path.basename(p)[:-4])
    return {**pose, "player": r3(np.median(pe)) if pe else None, "shuttle": r3(np.median(se)) if se else None,
            "shuttle_mean": r3(np.mean(se)) if se else None, "n_gt": len(se), "recall": round(rec, 2),
            "hits": len(hits), "labels": len(gt["hits"])}


def evaluate(model, verbose=False, full=False):
    res = []
    nan = lambda v: np.nan if v is None else v
    for p in eval_paths(full):
        e = evaluate_one(model, p)
        res.append((nan(e["player"]), nan(e["shuttle"]), e["recall"], e["hits"], e["labels"], nan(e["pose_rel"]), nan(e["arm_rel"])))
        if verbose:
            print(os.path.basename(p), "player %.3f shuttle %.3f recall %.2f hits %d/%d pose %.3f arm %.3f" % res[-1], flush=True)
    r = np.array(res, float)
    md = lambda c: round(float(np.nanmedian(r[:, c])), 3) if np.isfinite(r[:, c]).any() else None
    return {"rallies": len(r), "player_med": md(0), "shuttle_med": md(1),
            "hit_recall": round(float(np.nanmean(r[:, 2])), 3), "hits_per_label": round(float(r[:, 3].sum() / max(r[:, 4].sum(), 1)), 2),
            "pose_rel_med": md(5), "arm_rel_med": md(6), "pose_rallies": int(np.isfinite(r[:, 5]).sum())}


if __name__ == "__main__":
    if (sys.argv[1:] or ["train"])[0] == "train":
        train()
    else:
        m = load_model()
        print(evaluate(m, verbose=True, full=os.environ.get("FULL") == "1"))
