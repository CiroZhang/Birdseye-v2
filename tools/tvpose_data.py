"""Training / test arrays for the TV-only body + racket student (tools/tvpose_train.py).

Per rally and player slot, per TV frame:
  inputs (TV only): VIMO's 24 SMPL joints (TV view, ground-corrected placement; tools/fb_vimo.py), the TV camera, and
    the TV racket box nearest to the player's hands (YOLOv8x 'tennis racket', outputs/cctv/racket) expressed as metric
    offsets in the image plane at the hand's depth;
  targets (four-camera teacher): the pelvis of tools/body_mv.py (body_mvt2, where >= 2 cameras saw the player), and the
    racket of tools/racket_mv.py (racket_all1): grip relative to the body's hand, unit direction, racket hand.
The teacher's pose is VIMO's own, so only placement and racket are learned. No SMPL model or video is needed downstream.

usage: python tools/tvpose_data.py OUT.npz [--vimo vimo_mv|vimo_n4] [--uids file] [--no-target]
"""
import argparse, json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
from birdseye_v2.court import Camera  # noqa: E402
import body_mv as BM  # noqa: E402
import racket_mv as RM  # noqa: E402

SLOTS = ("bottom0", "bottom1", "top0", "top1")
NF = 24 * 3 + 3 + 3 + 6 + 7


def tv_cam(uid):
    return BM.static_cam(Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")))["cam"]))


def feats(J, cam, boxes, vfr):
    """J (T,24,3) VIMO joints -> (T,NF) features, (T,) valid."""
    T = len(J)
    F = np.zeros((T, NF), np.float32)
    ok = np.isfinite(J).all((1, 2))
    pel = J[:, 0]
    F[:, :72] = (J - pel[:, None]).reshape(T, 72)
    F[:, 72:75] = (pel - np.array([3.05, 6.7, 0.0])) / np.array([3.0, 6.7, 1.0])
    d = cam["C"][None] - pel
    F[:, 75:78] = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
    F[:, 78:84] = np.r_[cam["R"][0], cam["R"][1]][None]  # image right / down in court axes
    H, _ = RM.hand_points(np.where(ok[:, None, None], J, 0.0))  # (T,2,3)
    for i, f in enumerate(vfr):
        if not ok[i] or f >= len(boxes) or len(boxes[f]) == 0:
            continue
        D = boxes[f]; D = D[D[:, 4] > RM.BOX_CONF]
        if not len(D):
            continue
        uv, dep = BM.project(cam, H[i])                 # (2,2), (2,)
        if (dep < 0.3).any():
            continue
        m = cam["f"] / dep.mean()                         # px per metre at the hands
        c = 0.5 * (D[:, :2] + D[:, 2:4])
        dd = np.linalg.norm(c[:, None] - uv[None], axis=-1) / m  # (nb,2) metres
        b, h = np.unravel_index(dd.argmin(), dd.shape)
        if dd[b, h] > 0.9:
            continue
        F[i, 84:86] = (c[b] - uv[h]) / m
        F[i, 86:88] = (D[b, 2:4] - D[b, :2]) / m
        F[i, 88] = D[b, 4]
        F[i, 89] = h  # nearer hand: 0 left, 1 right
        F[i, 90] = 1.0
    F[~ok] = 0
    return F, ok


def targets(uid, slot, J, vfr):
    """teacher pelvis offset (T,3)+mask, grip offset from the input body's hand (T,3), direction (T,3), weight, hand."""
    T = len(J)
    out = dict(dp=np.zeros((T, 3), np.float32), dp_m=np.zeros(T, np.float32), grip=np.zeros((T, 3), np.float32),
               u=np.zeros((T, 3), np.float32), rk_w=np.zeros(T, np.float32), hand=np.float32(-1))
    b = np.load(os.path.join(ROOT, "outputs", "cctv", "body_mvt2", uid + ".npz"))
    if slot not in b.files:
        return out
    pos = {f: i for i, f in enumerate(b["frames"].astype(int))}
    idx = np.array([pos.get(f, -1) for f in vfr])
    have = idx >= 0
    Bt = np.full((T, 24, 3), np.nan); Bt[have] = b[slot][idx[have]]
    nobs = np.zeros((T, 4));
    if slot + "_nobs" in b.files:
        nobs[have] = b[slot + "_nobs"][idx[have]]
    okb = np.isfinite(Bt).all((1, 2)) & np.isfinite(J).all((1, 2))
    m = okb & ((nobs > 0).sum(1) >= 2)
    out["dp"][m] = Bt[m, 0] - J[m, 0]
    out["dp_m"][m] = 1.0
    rp = os.path.join(ROOT, "outputs", "cctv", "racket_all1", uid + ".npz")
    if os.path.exists(rp):
        r = np.load(rp)
        if slot + "_grip" in r.files:
            rpos = {f: i for i, f in enumerate(r["frames"].astype(int))}
            ri = np.array([rpos.get(f, -1) for f in vfr]); rh = ri >= 0
            G = np.full((T, 3), np.nan); U = np.full((T, 3), np.nan); nb = np.zeros((T, 4))
            G[rh] = r[slot + "_grip"][ri[rh]]; U[rh] = r[slot + "_u"][ri[rh]]; nb[rh] = r[slot + "_nbox"][ri[rh]]
            hand = int(r[slot + "_hand"]); hi = 1 if hand == 21 else 0
            Ht, _ = RM.hand_points(np.where(np.isfinite(Bt), Bt, 0.0))
            okr = okb & np.isfinite(G).all(1) & np.isfinite(U).all(1)
            out["grip"][okr] = G[okr] - Ht[okr, hi]
            out["u"][okr] = U[okr]
            out["rk_w"][okr] = np.where(nb[okr].sum(1) > 0, 1.0, 0.2)
            out["hand"] = np.float32(hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--vimo", default="vimo_mv"); ap.add_argument("--uids")
    ap.add_argument("--no-target", action="store_true")
    a = ap.parse_args()
    vd = os.path.join(ROOT, "outputs", "cctv", a.vimo)
    uids = open(a.uids).read().split() if a.uids else sorted(f[:-4] for f in os.listdir(vd))
    rows = {k: [] for k in ("uid", "slot", "X", "ok", "frames", "J", "dp", "dp_m", "grip", "u", "rk_w", "hand")}
    for n, uid in enumerate(uids):
        vp = os.path.join(vd, uid + ".npz")
        if not os.path.exists(vp):
            continue
        v = np.load(vp); vfr = v["frames"].astype(int)
        try:
            cam = tv_cam(uid)
            boxes = RM.load_boxes(os.path.join(ROOT, "outputs", "cctv", "racket", uid + ".npz"))
        except Exception as e:
            print(uid, "skip", e); continue
        for s in SLOTS:
            if s not in v.files or np.isfinite(v[s][:, 0, 0]).sum() < 16:
                continue
            J = v[s].astype(float)
            X, ok = feats(J, cam, boxes, vfr)
            T_ = len(J)
            t = targets(uid, s, J, vfr) if not a.no_target else dict(
                dp=np.zeros((T_, 3), np.float32), dp_m=np.zeros(T_, np.float32), grip=np.zeros((T_, 3), np.float32),
                u=np.zeros((T_, 3), np.float32), rk_w=np.zeros(T_, np.float32), hand=np.float32(-1))
            rows["uid"].append(uid); rows["slot"].append(s); rows["X"].append(X.astype(np.float16)); rows["ok"].append(ok)
            rows["frames"].append(vfr); rows["J"].append(J.astype(np.float32))
            for k in ("dp", "dp_m", "grip", "u", "rk_w", "hand"):
                rows[k].append(t[k])
        if n % 100 == 0:
            print(n, uid, len(rows["uid"]), flush=True)
    L = np.array([len(x) for x in rows["X"]])
    o = np.r_[0, np.cumsum(L)]
    pack = dict(uid=np.array(rows["uid"]), slot=np.array(rows["slot"]), off=o, hand=np.array(rows["hand"], np.float32))
    for k in ("X", "ok", "frames", "J", "dp", "dp_m", "grip", "u", "rk_w"):
        pack[k] = np.concatenate(rows[k])
    np.savez_compressed(a.out, **pack)
    print("saved", a.out, len(L), "tracks", o[-1], "frames")


if __name__ == "__main__":
    main()
