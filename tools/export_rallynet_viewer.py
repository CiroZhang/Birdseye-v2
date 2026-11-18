"""The demo viewer's data.json (same format as tools/export_viewer.py, so the same page renders it) from RallyNet
alone: skeletons, rackets, shuttle and hits all come out of the one model; the four detectors are its only input.
Ground truth (two-view shuttle, top-view players, labelled hits) is drawn for comparison only.

usage: python tools/export_rallynet_viewer.py BD15/Rally-39 ...   (env TAG=v0, VIEWER_OUT=viewer_rallynet)"""
import json, os, sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from birdseye_v2 import rallynet as RN  # noqa: E402
from birdseye_v2.court import NET_Y, Camera  # noqa: E402
from birdseye_v2.players import load_rackets  # noqa: E402
from birdseye_v2.tracks import RACKET_HEAD, RACKET_LEN  # noqa: E402
import train_rallynet as TR  # noqa: E402
from export_viewer import norm_px, r3, write_clip  # noqa: E402


def face_normals(w, u, full_speed=3.0):
    """Racket-face normal per frame, turning continuously: a swung racket moves face-first, so the head's velocity
    across the shaft says where the face points, trusted in proportion to the swing's speed (full at full_speed m/s)
    and otherwise carried along; filtered forward in time, sign-aligned (a face has two sides), always
    across the current shaft. Before the first swing it keeps the first swing's face."""
    T = len(u)
    head = np.nan_to_num(w + RN.GRIP_TO_HEAD * u)
    v = np.gradient(head, axis=0) * 30.0
    vp = v - np.sum(v * u, 1, keepdims=True) * u
    sp = np.linalg.norm(vp, axis=1)
    a = np.clip(sp / full_speed, 0, 1) ** 2 * 0.5
    d = vp / (sp[:, None] + 1e-9)
    ok = np.isfinite(u).all(1)

    def run(order):
        n, out = None, np.full((T, 3), np.nan)
        for i in order:
            if not ok[i]:
                continue
            if n is None:
                if a[i] < 0.1:
                    continue
                n = d[i]
            di = d[i] if d[i] @ n >= 0 else -d[i]
            m = (1 - a[i]) * n + a[i] * di
            m = m - (m @ u[i]) * u[i]
            if np.linalg.norm(m) > 1e-6:
                n = m / np.linalg.norm(m)
            out[i] = n
        return out
    f = run(range(T))
    b = run(range(T - 1, -1, -1))  # before the first swing: the first swing's face, carried back
    return np.where(np.isfinite(f), f, b)


def contact_rackets(S, wrist, u, pres, hits, reach=(0.2, 1.0), sd=2.5, k=6):
    """The racket meets the shuttle: at each fitted hit the hitter's racket (the present player whose racket head is
    nearest the fitted shuttle) turns to point at the shuttle, blended in and out over a few frames. Only the
    racket's direction changes -- the body is never moved by the shuttle."""
    u = u.copy()
    T = len(u)
    for a in hits:
        k0 = int(round(a))
        if not 0 <= k0 < T or not np.isfinite(S[k0]).all():
            continue
        X = S[k0]
        heads = wrist[k0] + RN.GRIP_TO_HEAD * u[k0]
        dist = np.where(pres[k0], np.linalg.norm(heads - X, axis=1), np.inf)
        q = int(np.argmin(dist))
        if not np.isfinite(dist[q]) or dist[q] > 1.2:
            continue
        for t in range(max(0, k0 - k), min(T, k0 + k + 1)):
            if not pres[t, q] or not np.isfinite(S[t]).all():
                continue
            v = S[k0] - wrist[t, q]  # the contact point, seen from the hand at frame t
            d = np.linalg.norm(v)
            if not reach[0] < d < reach[1]:
                continue
            g = np.exp(-0.5 * ((t - a) / sd) ** 2)
            m = (1 - g) * u[t, q] + g * v / d
            u[t, q] = m / (np.linalg.norm(m) + 1e-9)
    return u


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def _two_bone(sh, el0, target, a, b):
    """Elbow and wrist reaching from the shoulder towards target with bone lengths a, b; the elbow bends the way it
    already did (el0 is the pole). Out of reach: the arm straightens towards the target."""
    v = target - sh
    D = np.clip(np.linalg.norm(v), abs(a - b) + 1e-3, a + b - 1e-3)
    e = _unit(v)
    pole = (el0 - sh) - ((el0 - sh) @ e) * e
    if np.linalg.norm(pole) < 1e-4:
        pole = np.array([0, 0, -1.0]) - e[2] * e
    pole = _unit(pole)
    x = (a * a + D * D - b * b) / (2 * D)
    el = sh + x * e + np.sqrt(max(a * a - x * x, 0.0)) * pole
    return el, sh + D * e


def contact_swing(S, J, u, pres, hits, hand, sd=3.0, k=9, max_lean=0.35):
    """The hitter meets the shuttle: around each fitted hit the hitter (the present player whose racket head is
    nearest the fitted shuttle) swings the racket head onto it -- the target moves from the network's own racket
    head to the contact point and back (Gaussian in time), the racket lines up with the reach (shoulder to contact),
    and the arm is re-solved shoulder -> elbow -> wrist with its own bone lengths and its own elbow bend. A contact
    beyond the arm's reach leans the whole body towards it (at most max_lean m)."""
    J, u = J.copy(), u.copy()
    T, P = u.shape[:2]
    L = RN.GRIP_TO_HEAD
    swings = []
    for a_ in hits:
        k0 = int(round(a_))
        if not 0 <= k0 < T or not np.isfinite(S[k0]).all():
            continue
        X = S[k0]
        heads = J[k0, np.arange(P), 9 + hand] + L * u[k0]
        dist = np.where(pres[k0], np.linalg.norm(heads - X, axis=1), np.inf)
        q = int(np.argmin(dist))
        if not np.isfinite(dist[q]) or dist[q] > 3.0:
            continue
        sj, ej, wj = 5 + hand[q], 7 + hand[q], 9 + hand[q]
        ua = np.linalg.norm(J[k0, q, ej] - J[k0, q, sj])
        fa = np.linalg.norm(J[k0, q, wj] - J[k0, q, ej])
        reach = ua + fa + L
        short = max(np.linalg.norm(X - J[k0, q, sj]) - 0.97 * reach, 0.0)
        lean = min(short, max_lean) * _unit((X - J[k0, q, sj]) * np.array([1, 1, 0]))
        swings.append((q, a_))
        # the racket head swings THROUGH the contact along the shot's direction (backswing behind, follow-through after)
        k2 = min(k0 + 3, T - 1)
        vout = (S[k2] - X) * 30.0 / max(k2 - k0, 1) if np.isfinite(S[k2]).all() else np.zeros(3)
        vr = _unit(vout) * np.clip(0.4 * np.linalg.norm(vout), 4.0, 15.0) if np.linalg.norm(vout) > 1e-6 else np.zeros(3)
        for t in range(max(0, k0 - k), min(T, k0 + k + 1)):
            if not pres[t, q]:
                continue
            g = float(np.exp(-0.5 * ((t - a_) / sd) ** 2))
            J[t, q] = J[t, q] + g * lean
            sh, el0, wr0 = J[t, q, sj], J[t, q, ej], J[t, q, wj]
            H = wr0 + L * u[t, q]
            H = H + g * (X + vr * (t - a_) / 30.0 - H)
            d = _unit((1 - g) * u[t, q] + g * _unit(H - sh))
            el, wr = _two_bone(sh, el0, H - L * d, ua, fa)
            J[t, q, ej], J[t, q, wj] = el, wr
            u[t, q] = _unit(H - wr)
    return J, u, swings


def calm_arms(J, u, pres, hand, swings, sd=12.0, k=9, fade=6.0):
    """Only the hitter swings: away from a player's own hits (beyond +-k frames, fading in over `fade`) the racket
    arm and racket are smoothed in time (shoulder-relative, sd frames), so the network's fast swings at a shuttle
    someone else hits (a partner's smash flying past) become the slow arm motion of a player who is only watching."""
    J, u = J.copy(), u.copy()
    T, P = u.shape[:2]
    ts = np.arange(T)
    for q in range(P):
        keep = np.zeros(T)
        for q_, a_ in swings:
            if q_ == q:
                dt = np.maximum(np.abs(ts - a_) - k, 0.0)
                keep = np.maximum(keep, np.exp(-0.5 * (dt / fade) ** 2))
        on = pres[:, q] & np.isfinite(J[:, q]).all((1, 2))
        sj, ej, wj = 5 + hand[q], 7 + hand[q], 9 + hand[q]
        sh = J[:, q, sj]
        e_s = _smooth(np.where(on[:, None], J[:, q, ej] - sh, np.nan), sd)
        w_s = _smooth(np.where(on[:, None], J[:, q, wj] - sh, np.nan), sd)
        u_s = _smooth(np.where(on[:, None], u[:, q], np.nan), sd)
        for t in np.where(on)[0]:
            c = 1.0 - keep[t]
            if c < 1e-3 or not np.isfinite(w_s[t]).all():
                continue
            ua = np.linalg.norm(J[t, q, ej] - sh[t])
            fa = np.linalg.norm(J[t, q, wj] - J[t, q, ej])
            el0 = sh[t] + (1 - c) * (J[t, q, ej] - sh[t]) + c * e_s[t]
            tw = sh[t] + (1 - c) * (J[t, q, wj] - sh[t]) + c * w_s[t]
            J[t, q, ej], J[t, q, wj] = _two_bone(sh[t], el0, tw, ua, fa)
            u[t, q] = _unit((1 - c) * u[t, q] + c * u_s[t])
    return J, u


def _smooth(x, sd=1.5):
    """Gaussian smoothing along time of a (T,3) series (NaN-safe)."""
    T = len(x)
    k = np.arange(-int(3 * sd), int(3 * sd) + 1)
    w = np.exp(-0.5 * (k / sd) ** 2)
    out = np.full_like(x, np.nan)
    ok = np.isfinite(x).all(1)
    for t in range(T):
        i = t + k
        m = (i >= 0) & (i < T)
        m[m] &= ok[i[m]]
        if m.any():
            out[t] = (w[m, None] * x[i[m]]).sum(0) / w[m].sum()
    return out


def natural_heads(J, pres, pitch=(-5.0, 25.0), max_roll=15.0):
    """A head a neck can carry: the face is rigid (eyes and ears at fixed offsets from the nose), it nods between
    pitch degrees (back, forward) and rolls at most max_roll degrees from the trunk's axis -- the network leans far
    players' heads back ~30 deg, which no player does --, its facing is the network's (nose ahead of the neck, else
    the chest's normal), and it turns smoothly over time."""
    J = J.copy()
    T, P = J.shape[:2]
    tp = np.tan(np.radians(pitch))
    tr = np.tan(np.radians(max_roll))
    for q in range(P):
        on = pres[:, q] & np.isfinite(J[:, q]).all((1, 2))
        if not on.any():
            continue
        neck = (J[:, q, 5] + J[:, q, 6]) / 2
        hip = (J[:, q, 11] + J[:, q, 12]) / 2
        up = neck - hip
        up /= np.linalg.norm(up, axis=1, keepdims=True) + 1e-9
        lat = J[:, q, 5] - J[:, q, 6]  # right -> left shoulder
        lat = lat - np.sum(lat * up, 1, keepdims=True) * up
        lat /= np.linalg.norm(lat, axis=1, keepdims=True) + 1e-9
        chest = np.cross(lat, up)  # the way the chest faces
        v = J[:, q, 0] - neck
        fwd = v - np.sum(v * up, 1, keepdims=True) * up
        fwd = np.where((np.sum(fwd * chest, 1) < 0)[:, None], -fwd, fwd)  # the face is on the chest's side
        fwd = np.where((np.linalg.norm(fwd, axis=1) < 0.03)[:, None], chest, fwd)
        fwd = _smooth(np.where(on[:, None], fwd, np.nan))
        h = _smooth(np.where(on[:, None], v, np.nan))
        for t in np.where(on)[0]:
            u_, f_ = up[t], fwd[t] - (fwd[t] @ up[t]) * up[t]
            f_ = f_ / (np.linalg.norm(f_) + 1e-9)
            s_ = np.cross(u_, f_)
            hu = max(h[t] @ u_, 1e-3)  # nod and roll as slopes against the trunk axis, clipped to a neck's range
            hd = u_ + np.clip(h[t] @ f_ / hu, *tp) * f_ + np.clip(h[t] @ s_ / hu, -tr, tr) * s_
            hd = hd / np.linalg.norm(hd)
            ln = np.linalg.norm(np.cross(hd, f_))
            l_ = np.cross(hd, f_) / (ln + 1e-9) if ln > 1e-6 else lat[t]
            l_ = l_ if l_ @ lat[t] >= 0 else -l_  # keep left on the left
            fw = np.cross(l_, hd)  # face direction, perpendicular to the head axis
            fw = fw if fw @ f_ >= 0 else -fw
            nose = neck[t] + 0.21 * hd
            J[t, q, 0] = nose
            J[t, q, 1] = nose + 0.035 * l_ + 0.035 * hd - 0.02 * fw  # left eye
            J[t, q, 2] = nose - 0.035 * l_ + 0.035 * hd - 0.02 * fw  # right eye
            J[t, q, 3] = nose + 0.075 * l_ + 0.01 * hd - 0.09 * fw  # left ear
            J[t, q, 4] = nose - 0.075 * l_ + 0.01 * hd - 0.09 * fw  # right ear
    return J


OUT = os.path.join(ROOT, "outputs", os.environ.get("VIEWER_OUT", "viewer_rallynet"))
TAG = os.environ.get("TAG", "v0")
PRE, POST = 15, 30
GRIP = 0.08


def export(mr, model):
    m, r = mr.split("/")
    p = os.path.join(TR.OBS, "drone", f"{m}__{r}.npz")
    gt = json.load(open(p[:-4] + "_gt.json"))
    cam = Camera.from_dict(gt["cam"])
    f = TR.load_feats(p)
    out = TR.infer(model, f)
    T = len(f["C"])
    pres = f["present"] > 0
    for q in range(pres.shape[1]):  # a player the pose detector misses for a few frames is still there
        on = np.where(pres[:, q])[0]
        for a, b in zip(on[:-1], on[1:]):
            if 1 < b - a <= 30:
                pres[a:b, q] = True
    hp = 1 - np.prod(1 - 1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    hits = TR.hits_from(hp, nms=8) if TR.ARCH == "rallyphys" else TR.hits_from(hp)
    who = np.argmax(1 / (1 + np.exp(-out["hit"])) * f["present"], axis=1)
    g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
    bt_f = np.array(g["back_track"]["frames"], float)
    bt_px = np.array(g["back_track"]["px"], float).reshape(-1, 2)
    f0, f1 = max(0, int(bt_f[0]) - PRE), min(T - 1, int(bt_f[-1]) + POST)
    frames = np.arange(f0, f1 + 1)
    rk_path = os.path.join(ROOT, "outputs", "racket", m, r + ".npz")
    boxes_all = load_rackets(rk_path) if os.path.exists(rk_path) else []
    det = {int(round(a)): uv for a, uv in zip(bt_f, bt_px)}
    if TR.PHYS:
        S = TR.physics_shuttle(p, f, out)
    elif TR.ARCH == "rallyphys":  # as scored (tools/train_rallynet.evaluate_one)
        S, fit_shots = TR.physics_shuttle(p, f, out, return_shots=True)
    else:
        S = out["shuttle"]
    Jt = out["joints"] if TR.ARCH == "rallyphys" else RN.fix_head(out["joints"])
    # racket: the player's one racket hand for the rally + model direction
    if TR.ARCH == "rallyphys":  # the network's hand, one per player for the whole rally
        hand = (np.nanmean(np.where(pres, out["hand_p"], np.nan), 0) > 0.5).astype(int)
        u = out["u2"][:, np.arange(RN.P), hand]
        u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    else:
        hand = RN.racket_hand(out["joints"], f)
        u = out["racket_u"]
    Jt = natural_heads(Jt, pres)
    if TR.ARCH == "rallyphys" and fit_shots:
        # a swing only at the network's hits: the other shot boundaries are flights split where the track turns
        # (SPLIT_GIVEN), not a player's stroke -- swinging there makes a partner hit the shuttle he only watched
        real = [a for a, _ in fit_shots[1:] if len(hits) and np.min(np.abs(np.asarray(hits, float) - a)) <= 4]
        Jt, u, swings = contact_swing(S, Jt, u, pres, real, hand)  # the first shot starts with the track
        Jt, u = calm_arms(Jt, u, pres, hand, swings)
        if os.environ.get("SWING_DEBUG"): print(mr, "hitters", [(q, round(float(a), 1)) for q, a in swings], flush=True)
    wrist = Jt[:, np.arange(Jt.shape[1]), 9 + hand]
    gt_by = {}
    for t, *X in gt["shuttle"]:
        k = int(round(t))
        if abs(t - k) <= 0.5:
            gt_by[k] = X
    per = []
    normals = [face_normals(wrist[frames, q], u[frames, q]) for q in range(RN.P)]
    for i, fr in enumerate(frames):
        P, Rk = [], []
        for q in range(RN.P):
            if not pres[fr, q]:
                continue
            X = Jt[fr, q]
            P.append({"id": q, "s": "n" if X[[15, 16], 1].mean() < NET_Y else "f", "j": [r3(x) for x in X],
                      "k": [norm_px(x, cam.size) for x in cam.project(X)]})
            w_, u_ = wrist[fr, q], u[fr, q]
            butt, tip = w_ - GRIP * u_, w_ + (RACKET_LEN - GRIP) * u_
            nv = normals[q][i]
            Rk.append({"id": q, "w": r3(w_), "u": r3(u_), "seen": True, **({"n": r3(nv)} if np.isfinite(nv).all() else {}),
                       "px": [norm_px(x, cam.size) for x in cam.project(np.array([butt, tip]))]})
        bx = boxes_all[fr] if fr < len(boxes_all) else np.zeros((0, 5))
        per.append({"p": P, "r": Rk, "rb": [norm_px(b[:2], cam.size) + norm_px(b[2:4], cam.size) for b in bx if b[4] >= 0.25],
                    "x": r3(S[fr]), "xp": norm_px(cam.project(S[fr][None])[0], cam.size),
                    "d": norm_px(det[fr], cam.size) if fr in det else None,
                    "g": r3(np.array(gt_by[fr])) if fr in gt_by else None,
                    "pg": [[round(a, 2), round(b, 2)] for a, b in gt["players"].get(str(fr), [])]})
    ck = lambda t: round(float(t) - f0, 2)
    shots = []
    hs = [h for h in hits if f0 <= h <= f1]
    for k, h in enumerate(hs):
        e = hs[k + 1] if k + 1 < len(hs) else min(f1, h + 60)
        seg = S[h:e + 1]
        v = np.linalg.norm(np.diff(seg[:6], axis=0), axis=1) * float(1 / f["dt"]) if len(seg) > 2 else [0]
        q = who[h]
        shots.append({"t0": ck(h), "t1": ck(e), "side": "near" if Jt[h, q, [15, 16], 1].mean() < NET_Y else "far",
                      "speed": float(np.max(v) * 3.6), "apex": float(seg[:, 2].max()), "anchored": True,
                      "path": [r3(x) for x in seg]})
    ev = TR.evaluate_one(model, p, out=out, S=S)
    R = cam.R
    clip = f"{m}_{r}.mp4"
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(os.path.join(OUT, clip)):
        write_clip(os.path.join(ROOT, "data", "videos", m, r, "(1).mp4"), os.path.join(OUT, clip), f0, f1)
    meta = {"id": f"{m} {r}", "clip": clip, "skel": "coco17", "n": len(frames), "f0": int(f0),
            "cam": {"c": r3(cam.center), "fwd": r3(R.T @ [0, 0, 1.0]), "down": r3(R.T @ [0, 1.0, 0]),
                    "vfov": round(float(np.degrees(2 * np.arctan(cam.size[1] / 2 / cam.f))), 2),
                    "aspect": round(cam.size[0] / cam.size[1], 4)},
            "shots": shots, "labels": [ck(l) for l in gt["hits"] if f0 <= l <= f1],
            "gt_path": [[ck(t)] + [round(float(x), 3) for x in X] for t, *X in gt["shuttle"]],
            "racket": {"len": RACKET_LEN, "head": list(RACKET_HEAD), "grip": GRIP},
            "metrics": {"player_err_median": ev["player"], "err_median": ev["shuttle"], "err_mean": ev["shuttle_mean"],
                        "n_gt": ev["n_gt"], "hits_detected": len(hs), "hits_labeled": len(gt["hits"]),
                        "hit_recall": ev["recall"], "has_racket": True},
            "frames": per}
    print(mr, meta["metrics"], flush=True)
    return meta


if __name__ == "__main__":
    model = TR.load_model(TAG)
    data = [export(mr, model) for mr in sys.argv[1:]]
    json.dump(data, open(os.path.join(OUT, "data.json"), "w"), separators=(",", ":"))
    print("wrote", os.path.join(OUT, "data.json"), os.path.getsize(os.path.join(OUT, "data.json")) // 1024, "KB")
