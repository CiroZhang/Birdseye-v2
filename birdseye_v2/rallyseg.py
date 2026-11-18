"""Hits and 3D shuttle flights from ONE objective: the rally is a chain of drag+gravity flights between
hits, each hit is at a player's hand, and hitters alternate court sides. The hit times are latent; they
are chosen jointly with the flights by dynamic programming over candidate times, instead of being
detected first (shuttle3d.detect_hits / hitclf) and then fitted.

  nodes      the track start, every candidate hit time (image-space turns > 15 deg, plus times inside
             detection gaps -- the shuttle is often hidden at contact), and the track end (landing,
             or last detection). A candidate node has a court side: its anchor is that side's player's
             hand nearest the shuttle in the image at that time (3D: YOLO-pose lift or VIMO body).
  flight     cost(i, s -> j) = least-squares cost of the best single flight from node i to node j:
             robust pixel residuals of the detections in between, start at node i's hand, end at node
             j's hand (the next hitter, other side) or above the rest point (landed end), floor, net
             tape cleared. Parameters (p0, v0) only -- the same physics as shuttle3d.RallyFit.
  rally      sum of flight costs + LAMBDA per hit; the DP over (node, side) finds the best chain exactly.

segment(cam, frames, px, anchors_at, ...) -> {"hits": [t], "sides": [0 near / 1 far], "hands": [3D], "cost"}
anchors_at(t) -> [{"hand": 3D, "side": 0|1, "px": image distance}] (eval_shuttle.hand_cands_for).
"""
import numpy as np
from scipy.optimize import least_squares

from .court import NET_Y, POST_H, L
from .hitclf import _candidates
from .shuttle3d import FPS, N_GRID, _ray_point, landing_frame, rest_point, simulate

LAMBDA = 8.0  # cost per hit (in units of 0.5 * squared normalised residual)
MAX_FLIGHT = 100  # frames; longest flight considered (a high clear / lift is ~2 s = 60 frames)
MIN_FLIGHT = 3
GAP_STEP = 3.0  # frames between candidates placed inside a detection gap


def candidates(frames, px, angle_deg=15.0):
    c = [x[1] for x in _candidates(frames, px, angle_deg=angle_deg, nms=3)]
    for a, b in zip(frames[:-1], frames[1:]):
        if b - a > 3:
            c += list(np.arange(a + 2, b - 1, GAP_STEP))
    c = sorted(set(round(float(t), 2) for t in c if frames[0] + 1 < t < frames[-1] - 1))
    return c


def _robust(r):
    return np.sign(r) * np.sqrt(2 * (np.sqrt(1 + r * r) - 1))


def flight(cam, f, u, ta, tb, A=None, B=None, rest=None, sigma_px=4.0, sigma_hand=0.6, sigma_floor=0.05,
           sigma_net=0.03, net_margin=0.1, sigma_land=0.15, cross=True):
    """Best single flight from time ta to tb -> (cost, p0, v0). A / B: 3D start / end anchors (hands);
    rest: floor point the flight comes down above (landed end). cross: the flight crosses the net, so it
    must clear the tape."""
    T = (tb - ta) / FPS
    ts = np.r_[(f - ta) / FPS, T, np.linspace(0, T, N_GRID)]
    p_start = np.asarray(A, float) if A is not None else (_ray_point(cam, u[0], 1.5) if len(f) else np.array([3.05, L / 2, 1.5]))
    if B is not None:
        p_end = np.asarray(B, float)
    elif rest is not None:
        p_end = np.asarray(rest, float)
    elif len(f):
        p_end = _ray_point(cam, u[-1], 0.5)
    else:
        p_end = p_start + np.array([0, 4.0 if p_start[1] < NET_Y else -4.0, 0])
    v = (p_end - p_start) / max(T, 0.05)
    v[2] += 0.5 * 9.81 * T
    try:
        v = least_squares(lambda vv: simulate(p_start, vv, [T])[0] - p_end, v, max_nfev=30).x
    except Exception:
        pass
    n = len(f)

    def res(x):
        X = simulate(x[:3], x[3:], ts)
        out = []
        if n:
            out.append(_robust(((cam.project(X[:n]) - u) / sigma_px).ravel()))
        if A is not None:
            out.append((x[:3] - A) / sigma_hand)
        if B is not None:
            out.append((X[n] - B) / sigma_hand)
        if rest is not None:
            out.append((X[n, :2] - rest[:2]) / sigma_land)
        G = X[n + 1:]
        out.append(np.minimum(G[:, 2], 0) / sigma_floor)
        if cross:
            dy = G[:, 1] - NET_Y
            j = np.flatnonzero(dy[:-1] * dy[1:] <= 0)
            c = 0.0
            if len(j):
                w = np.where(dy[j] != dy[j + 1], dy[j] / (dy[j] - dy[j + 1] + 1e-12), 0.0)
                c = min(np.min((1 - w) * G[j, 2] + w * G[j + 1, 2]) - POST_H - net_margin, 0.0)
            out.append(np.atleast_1d(c / sigma_net))
        return np.concatenate(out)

    x0 = np.r_[p_start, v]
    try:
        r = least_squares(res, x0, x_scale="jac", max_nfev=60)
        return float(r.cost), r.x[:3], r.x[3:]
    except Exception:
        return float("inf"), x0[:3], x0[3:]


def segment(cam, frames, px, anchors_at, lam=LAMBDA, land=True, max_flight=MAX_FLIGHT, unary=None, **kw):
    """DP over (node, side). unary(t) -> extra cost of a hit at t (e.g. -log p from a hit classifier)."""
    frames, px = np.asarray(frames, float), np.asarray(px, float)
    lf = landing_frame(frames, px) if land else None
    rest = rest_point(cam, frames, px, lf) if lf is not None else None
    t_end = float(lf) if lf is not None else float(frames[-1])
    cands = [t for t in candidates(frames, px) if t < t_end - 2]
    t0 = float(frames[0])
    times = [t0] + cands + [t_end]
    K = len(times)
    # anchors per (node, side); the start node may also be unanchored (flight already under way)
    anc = {}
    for k in range(K - 1):
        cs = anchors_at(times[k])
        for s in (0, 1):
            same = [c for c in cs if c["side"] == s]
            anc[k, s] = same[0]["hand"] if same else None
    INF = float("inf")
    # start states: side 0 / 1 = the track starts at that side's hit (serve); 2 = already in flight (no anchor)
    best = {(0, s): (0.0, None) for s in (0, 1, 2)}  # (cost, back-pointer (node, side))
    cache = {}

    def obs(ta, tb):
        m = (frames > ta) & (frames < tb)
        return frames[m], px[m]

    for j in range(1, K):
        last = j == K - 1
        for sj in ((0,) if last else (0, 1)):
            if not last and anc[j, sj] is None:
                continue
            cand = (INF, None)
            for i in range(j - 1, -1, -1):
                if times[j] - times[i] > max_flight:
                    break
                if times[j] - times[i] < MIN_FLIGHT:
                    continue
                for si in ((0, 1, 2) if i == 0 else (0, 1)):
                    if (i, si) not in best or best[i, si][0] == INF:
                        continue
                    if not last and si == sj:
                        continue  # hitters alternate sides
                    A = anc.get((i, si))  # None for the unanchored start
                    B = None if last else anc[j, sj]
                    key = (i, si, j, sj)
                    if key not in cache:
                        f, u = obs(times[i], times[j])
                        cache[key] = flight(cam, f, u, times[i], times[j], A=A, B=B,
                                            rest=rest if last else None, cross=not last, **kw)[0]
                    c = best[i, si][0] + cache[key] + (0.0 if last else lam + (unary(times[j]) if unary else 0.0))
                    if c < cand[0]:
                        cand = (c, (i, si))
            best[j, sj] = cand
    # back-track
    node = (K - 1, 0)
    if best[node][1] is None:
        return {"hits": [], "sides": [], "hands": [], "cost": INF, "n_cands": len(cands)}
    chain = []
    while node is not None:
        chain.append(node)
        node = best[node][1]
    chain = chain[::-1]
    inner = chain[1:-1]
    return {"hits": [times[k] for k, _ in inner], "sides": [s for _, s in inner],
            "hands": [anc[k, s] for k, s in inner], "start_side": chain[0][1], "cost": best[K - 1, 0][0],
            "n_cands": len(cands), "n_fits": len(cache)}
