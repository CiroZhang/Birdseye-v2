"""Multi-camera rackets: each player's racket as a rigid stick in the hand, fitted to every camera's racket boxes.

The racket is the grip G_k (the racket hand of the multi-camera body, tools/body_mv.py, 7 cm past the wrist) and a
unit direction u_k: head centre G + HEAD_D u, tip G + TIP_D u, head radius HEAD_R. Per player the direction track is
fitted (robust least squares) to
  - racket boxes (YOLOv8x COCO 'tennis racket') of the TV, side and player cameras: the projected racket's box
    (grip, tip and the head disc) against each detected box matched to this player,
  - smooth motion (second difference of u),
  - a weak prior to the forearm direction (elbow -> wrist), which holds the racket where nothing sees it,
  - contact: at each teacher hit, the hitter's head centre at the shuttle (RK_HIT=0 turns it off, for the check).
The racket hand of each player is the wrist its matched boxes are nearest to (a vote over the rally).
Output outputs/cctv/<RK_DIR or racket_mv>/<uid>.npz: per slot <s>_grip, <s>_u (T,3), <s>_hand (20|21), <s>_nbox (T,4),
<s>_hits (TV times of this player's hits).
usage: python tools/racket_mv.py uid ...   |   python tools/racket_mv.py --eval uid ...  (hits held out)"""
import json, os, sys

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.sparse import lil_matrix

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import body_mv as BM  # noqa: E402

HEAD_D, TIP_D, HEAD_R, HAND = 0.47, 0.60, 0.11, 0.07  # m (badminton racket 0.665 m, head 0.23 x 0.20)
SIG_BOX = {"tv": 4.0, "side": 4.0, "p1": 12.0, "p2": 12.0}  # px, plus 15% of the box size
SIG_SMOOTH = 0.25     # second difference of the unit direction per frame
SIG_ARM = 0.8         # deviation from the forearm direction (unit-vector difference, ~50 deg)
SIG_HIT = 0.04        # m: head centre to shuttle at a hit
SIG_GRIP = float(os.environ.get("RK_SIG_GRIP", "0.10"))  # m: grip away from the body's hand
SIG_GRIP_ACC = 0.03   # m/frame^2: smoothness of that offset
BOX_CONF = 0.3
SLOTS = BM.SLOTS
OUT = os.path.join(ROOT, "outputs", "cctv", os.environ.get("RK_DIR", "racket_mv"))
TEACH = os.environ.get("RK_TEACH", os.path.join(ROOT, "outputs", "cctv", "teach"))
BODY = os.environ.get("RK_BODY", "body_mv")
HIT = os.environ.get("RK_HIT", "1") == "1"


def load_boxes(p):
    z = np.load(p); c = np.r_[0, np.cumsum(z["counts"])]
    return [z["boxes"][c[i]:c[i + 1]] for i in range(len(z["counts"]))]


def box_cameras(uid, drop=()):
    """[(view, frame->camera, frame->TV time, boxes per frame)] from the pose cameras of body_mv."""
    out = []
    for v, camf, tvt, _ in BM.cameras(uid):
        base = {"tv": "cctv", "side": "cctv_side"}.get(v, "cctv_" + v)
        p = os.path.join(ROOT, "outputs", base, "racket", uid + ".npz")
        if os.path.exists(p) and v not in drop:
            out.append((v, camf, tvt, load_boxes(p)))
    return out


def hand_points(Jw):
    """(T,2,3) hand centres (left, right) and (T,2,3) unit forearm directions from SMPL joints."""
    W, E = Jw[:, [20, 21]], Jw[:, [18, 19]]
    f = W - E; f /= np.linalg.norm(f, axis=-1, keepdims=True) + 1e-9
    return W + HAND * f, f


def racket_pts(G, u):
    return G, G + TIP_D * u, G + HEAD_D * u


def proj_box(cam, G, u):
    """Projected racket box (...,4) x0 y0 x1 y1 and depth ok mask."""
    P = np.stack(racket_pts(G, u), -2)        # (...,3,3)
    uv, dep = BM.project(cam, P)
    rho = cam["f"] * HEAD_R / np.maximum(dep[..., 2], 0.3)
    hx, hy = uv[..., 2, 0], uv[..., 2, 1]
    x0 = np.minimum(np.minimum(uv[..., 0, 0], uv[..., 1, 0]), hx - rho)
    x1 = np.maximum(np.maximum(uv[..., 0, 0], uv[..., 1, 0]), hx + rho)
    y0 = np.minimum(np.minimum(uv[..., 0, 1], uv[..., 1, 1]), hy - rho)
    y1 = np.maximum(np.maximum(uv[..., 0, 1], uv[..., 1, 1]), hy + rho)
    return np.stack([x0, y0, x1, y1], -1), (dep > 0.3).all(-1)


def gather(cams, H, F, U, vfr):
    """Match each camera frame's racket boxes to the players' rackets.
    H (B,T,2,3) hands, U (B,T,3) current directions (or None: match to either hand) -> obs list."""
    pos = {f: i for i, f in enumerate(vfr)}
    obs = []
    B = len(H)
    for vi, (v, camf, tvt, K) in enumerate(cams):
        for k, D in enumerate(K):
            D = D[D[:, 4] > BOX_CONF] if len(D) else D
            if len(D) == 0:
                continue
            cam = camf(k)
            if cam is None:
                continue
            t = tvt(k); f0 = int(np.floor(t)); a = t - f0
            if f0 in pos and f0 + 1 in pos:
                i0, i1 = pos[f0], pos[f0 + 1]
            elif f0 in pos and a < 1e-6:
                i0 = i1 = pos[f0]
            else:
                continue
            cost = np.full((B, len(D)), 1e9); side = np.zeros((B, len(D)), int)
            for b in range(B):
                h = (1 - a) * H[b, i0] + a * H[b, i1]
                if not np.isfinite(h).all():
                    continue
                uvh, dep = BM.project(cam, h)
                if (dep < 0.3).any():
                    continue
                reach = cam["f"] * 0.8 / dep.min()  # px of 0.8 m at the hand's depth
                c = 0.5 * (D[:, :2] + D[:, 2:4])
                for d in range(len(D)):
                    # box to the hand: distance from the hand to the box (0 inside), per hand
                    dx = np.maximum(np.maximum(D[d, 0] - uvh[:, 0], uvh[:, 0] - D[d, 2]), 0)
                    dy = np.maximum(np.maximum(D[d, 1] - uvh[:, 1], uvh[:, 1] - D[d, 3]), 0)
                    e = np.hypot(dx, dy) + 0.05 * np.linalg.norm(c[d] - uvh, axis=1)
                    j = int(np.argmin(e))
                    if e[j] < max(reach, 15.0):
                        cost[b, d], side[b, d] = e[j] / max(reach, 15.0), j
            for b, d in zip(*linear_sum_assignment(cost)):
                if cost[b, d] < 1e9:
                    obs.append((b, vi, i0, i1, a, cam, D[d, :4], side[b, d]))
    return obs


def soft_l1(r, f):
    """Residuals whose squares give the soft-L1 cost (robust to wrong boxes), applied to one block only."""
    q = r / f
    return f * np.sign(q) * np.sqrt(2 * (np.sqrt(1 + q * q) - 1))


def cauchy(r, f):
    """Residuals whose squares give the Cauchy cost: an unreachable hit gives up instead of dragging the racket."""
    return f * np.sign(r) * np.sqrt(np.log1p((r / f) ** 2))


def fit_racket(G, Fd, obs_b, sig, hits):
    """(grip (T,3), direction (T,3)) for one player.  G (T,3) body's hand, Fd (T,3) forearm direction.
    The grip may leave the body's hand by a smooth offset (the body is off by its own error; SIG_GRIP)."""
    T = len(G)
    idx = np.flatnonzero(np.isfinite(G).all(1) & np.isfinite(Fd).all(1))
    var = {f: j for j, f in enumerate(idx)}
    n = len(idx)
    ob = [o for o in obs_b if o[1] in var and o[2] in var]
    hits = [h for h in hits if h[0] in var and h[1] in var]
    acc = np.array([(var[f - 1], var[f], var[f + 1]) for f in idx if f - 1 in var and f + 1 in var], int).reshape(-1, 3)
    # residual blocks: (rows, frames they depend on)
    blocks = [(4, (var[o[1]], var[o[2]])) for o in ob] + [(3, (j,)) for j in range(n)] * 2 + [(1, (j,)) for j in range(n)]         + [(3, tuple(t)) for t in acc] * 2 + [(3, (var[h[0]], var[h[1]])) for h in hits]
    J = lil_matrix((sum(b[0] for b in blocks), 6 * n), dtype=np.int8)
    r = 0
    for nr, fs in blocks:
        for j in fs:
            J[r:r + nr, 6 * j:6 * j + 6] = 1
        r += nr
    Gi, Fi = G[idx], Fd[idx]
    groups = {}
    for q, o in enumerate(ob):
        groups.setdefault(id(o[4]), (o[4], []))[1].append(q)
    Gr = []
    for cam, qs in groups.values():
        o = [ob[q] for q in qs]
        j0 = np.array([var[x[1]] for x in o]); j1 = np.array([var[x[2]] for x in o]); a = np.array([x[3] for x in o])
        box = np.array([x[5] for x in o]); s = np.array([sig[x[0]] for x in o])
        s = s[:, None] + 0.15 * np.c_[box[:, 2] - box[:, 0], box[:, 3] - box[:, 1]].max(1)[:, None]
        Gr.append((cam, np.array(qs), j0, j1, a, box, s))
    hj = np.array([(var[h[0]], var[h[1]], h[2]) for h in hits]).reshape(-1, 3)
    hP = np.array([h[3] for h in hits]).reshape(-1, 3)

    def unit(w):
        return w / (np.linalg.norm(w, axis=-1, keepdims=True) + 1e-9)

    def at(x, j0, j1, a):
        return (1 - a)[:, None] * x[j0] + a[:, None] * x[j1]

    def res(p):
        p = p.reshape(n, 6); w, dg = p[:, :3], p[:, 3:]
        u = unit(w); g = Gi + dg
        e = np.zeros((len(ob), 4))
        for cam, qs, j0, j1, a, box, s in Gr:
            pb, _ = proj_box(cam, at(g, j0, j1, a), unit(at(u, j0, j1, a)))
            e[qs] = (pb - box) / s
        out = [soft_l1(e.ravel(), 2.0), ((u - Fi) / SIG_ARM).ravel(), (dg / SIG_GRIP).ravel(), (np.linalg.norm(w, axis=1) - 1) / 0.1]
        if len(acc):
            out.append(((u[acc[:, 0]] - 2 * u[acc[:, 1]] + u[acc[:, 2]]) / SIG_SMOOTH).ravel())
            out.append(((dg[acc[:, 0]] - 2 * dg[acc[:, 1]] + dg[acc[:, 2]]) / SIG_GRIP_ACC).ravel())
        if len(hj):
            j0, j1, a = hj[:, 0].astype(int), hj[:, 1].astype(int), hj[:, 2]
            e = ((at(g, j0, j1, a) + HEAD_D * unit(at(u, j0, j1, a)) - hP) / SIG_HIT).ravel()
            out.append(e if os.environ.get("RK_HIT_HARD") == "1" else cauchy(e, 3.0))   # hard: the head is on the shuttle
        return np.concatenate(out)

    sol = least_squares(res, np.c_[Fi, np.zeros_like(Fi)].ravel(), jac_sparsity=J, max_nfev=100)
    p = sol.x.reshape(n, 6)
    U = np.full((T, 3), np.nan); Gout = np.full((T, 3), np.nan)
    U[idx] = unit(p[:, :3]); Gout[idx] = Gi + p[:, 3:]
    return Gout, U


def teacher(uid):
    out = json.load(open(os.path.join(TEACH, uid + ".json")))
    fr = np.array(out["frames"], float); P = np.array(out["P"], float)
    Pat = lambda t: np.array([np.interp(t, fr, P[:, c]) for c in range(3)])
    return [h for h in out["hits"] if fr[0] <= h <= fr[-1]], Pat


def decoded_hits(uid):
    """RK_HITS=<json> {uid: [[frame, multi-camera track]]} (tools/hit_decode.py HD_HITS): hits with their hitter's
    body slot (named as tools/fb_vimo.py mv_boxes does); only hits where the teacher is sure of the shuttle"""
    m = np.load(os.path.join(ROOT, "outputs", "cctv", "mv_track", uid + ".npz")); K = int(m["K"])
    side = [np.nanmedian(m["foot"][:, k, 1]) > 6.7 for k in range(K)]
    name = {k: ("top" if side[k] else "bottom") + str(sum(1 for j in range(k) if side[j] == side[k])) for k in range(K)}
    if os.environ.get("RK_CONTACTS"):                  # shot fit (tools/shot_fit.py): every hit's contact point
        sf = json.load(open(os.path.join(os.environ["RK_CONTACTS"], uid + ".json")))
        return [(float(f), name[k], np.array([x, y, z])) for f, k, x, y, z in sf["contacts"]]
    out = json.load(open(os.path.join(TEACH, uid + ".json")))
    fr = np.array(out["frames"], float); sd = np.array(out["sd"], float)
    return [(float(f), name[k], None) for f, k in json.load(open(os.environ["RK_HITS"])).get(uid, [])
            if fr[0] <= f <= fr[-1] and np.interp(f, fr, sd) < 0.06]


def run(uid, drop=(), use_hits=HIT):
    vb = np.load(os.path.join(ROOT, "outputs", "cctv", BODY, uid + ".npz"))
    vfr = vb["frames"].astype(int); pos = {f: i for i, f in enumerate(vfr)}
    names = [s for s in SLOTS if s in vb.files and np.isfinite(vb[s][:, 0, 0]).sum() >= 10]
    Hh, Ff = zip(*[hand_points(vb[s].astype(float)) for s in names])
    Hh, Ff = np.array(Hh), np.array(Ff)        # (B,T,2,3)
    cams = box_cameras(uid, drop)
    sig = [SIG_BOX[c[0]] for c in cams]
    obs = gather(cams, Hh, Ff, None, vfr)
    hits, Pat = teacher(uid)
    res = {"frames": vb["frames"]}
    for b, s in enumerate(names):
        ob = [o for o in obs if o[0] == b]
        hand = 1 if sum(o[7] for o in ob) >= len(ob) / 2 else 0   # racket hand: 0 left (SMPL 20), 1 right (21)
        G, Fd = Hh[b, :, hand], Ff[b, :, hand]
        hb = []
        if use_hits and os.environ.get("RK_HITS"):
            for h, slot, Cp in decoded_hits(uid):
                f0 = int(np.floor(h)); a = h - f0
                if slot == s and f0 in pos and f0 + 1 in pos:
                    hb.append((pos[f0], pos[f0 + 1], a, Cp if Cp is not None else Pat(h)))
        elif use_hits:
            for h in hits:
                f0 = int(np.floor(h)); a = h - f0
                if f0 not in pos or f0 + 1 not in pos:
                    continue
                P = Pat(h)
                # the hitter: the player whose racket hand is nearest the shuttle (within reach)
                d = [np.linalg.norm((1 - a) * Hh[c, pos[f0], hd] + a * Hh[c, pos[f0 + 1], hd] - P)
                     if np.isfinite(Hh[c, [pos[f0], pos[f0 + 1]]]).all() else 9 for c in range(len(names)) for hd in (0, 1)]
                if int(np.argmin(d)) // 2 == b and min(d) < 1.6:
                    hb.append((pos[f0], pos[f0 + 1], a, P))
        G, U = fit_racket(G, Fd, [o[1:7] for o in ob], sig, hb)
        res[s + "_hits"] = np.array([vfr[h[0]] + h[2] for h in hb], float)  # TV times where this player hits
        res[s + "_grip"] = G.astype(np.float32); res[s + "_u"] = U.astype(np.float32); res[s + "_hand"] = 20 + hand
        nb = np.zeros((len(vfr), 4), np.int16)
        for o in ob:
            nb[o[2], BM.VIEWS.index(cams[o[1]][0])] += 1
        res[s + "_nbox"] = nb
    return res, cams, names


def evaluate(uid):
    """Hits held out: racket head centre to the teacher's shuttle at each hit (hitter = nearest racket head),
    and held-out-camera box agreement (IoU) for each player camera."""
    hits, Pat = teacher(uid)
    out = {}
    res, _, names = run(uid, use_hits=False)
    vfr = res["frames"].astype(int); pos = {f: i for i, f in enumerate(vfr)}
    dh = []
    for h in hits:
        f = int(round(h))
        if f not in pos:
            continue
        P = Pat(h)
        hd = [np.linalg.norm(res[s + "_grip"][pos[f]] + HEAD_D * res[s + "_u"][pos[f]] - P) for s in names]
        hd = [x for x in hd if np.isfinite(x)]
        if hd:
            dh.append(min(hd))
    out["hit_head_m"] = dh
    for v in ("p1", "p2", "side"):
        r2, cams, names = run(uid, drop=(v,), use_hits=False)
        test = [c for c in box_cameras(uid) if c[0] == v]
        if not test:
            continue
        Hh = np.array([hand_points(np.load(os.path.join(ROOT, "outputs", "cctv", BODY, uid + ".npz"))[s].astype(float))[0] for s in names])
        iou = []
        for (b, vi, i0, i1, a, cam, box, sd) in gather(test, Hh, None, None, vfr):
            s = names[b]
            G = (1 - a) * r2[s + "_grip"][i0] + a * r2[s + "_grip"][i1]
            u = (1 - a) * r2[s + "_u"][i0] + a * r2[s + "_u"][i1]; u /= np.linalg.norm(u) + 1e-9
            if not (np.isfinite(G).all() and np.isfinite(u).all()):
                continue
            pb, ok = proj_box(cam, G, u)
            x0, y0 = max(pb[0], box[0]), max(pb[1], box[1]); x1, y1 = min(pb[2], box[2]), min(pb[3], box[3])
            inter = max(x1 - x0, 0) * max(y1 - y0, 0)
            uni = (pb[2] - pb[0]) * (pb[3] - pb[1]) + (box[2] - box[0]) * (box[3] - box[1]) - inter
            iou.append(inter / max(uni, 1e-9))
        out["iou_" + v] = iou
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--eval":
        A = {}
        for uid in args[1:]:
            r = evaluate(uid)
            for k, v in r.items():
                A.setdefault(k, []).extend(v)
            print(uid, " ".join(f"{k} n={len(v)} med {np.median(v):.2f}" for k, v in r.items() if v), flush=True)
        for k, v in A.items():
            v = np.array(v)
            extra = f" within 0.15 m {np.mean(v < 0.15):.0%} within 0.3 m {np.mean(v < 0.3):.0%}" if k.startswith("hit") else f" IoU>0.5 {np.mean(v > 0.5):.0%}"
            print(f"ALL {k}: n={len(v)} med {np.median(v):.2f}" + extra)
        sys.exit()
    os.makedirs(OUT, exist_ok=True)
    for uid in args:
        res, cams, names = run(uid)
        np.savez_compressed(os.path.join(OUT, uid + ".npz"), **res)
        print(uid, "players", names, "boxes", {c[0]: int(sum(res[s + "_nbox"][:, BM.VIEWS.index(c[0])].sum() for s in names))
                                                for c in cams}, flush=True)
