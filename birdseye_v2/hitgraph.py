"""Global start for the dense teacher: the rally as a chain of hits, found by dynamic programming.

Where only the TV sees the shuttle, its rays are nearly horizontal and carry almost no depth; the dense solve then
settles in whatever depth its start suggests (measured: started from the two-view truth it ends at a far lower cost,
so the objective is right and the search is not). Here the same model is searched globally instead:
  nodes  -- candidate hits: points on a camera ray within REACH of a player's feet (and below MAX_HIT_Z), plus
            landings (ray meets the floor);
  edges  -- the drag flight between two nodes (boundary-value problem, solved by Newton shooting), scored by the
            robust pixel cost of every fixed camera's detections along it (Cauchy, as in the dense solve);
  chain  -- a second-order DP over edges: a hit costs log(1 + (dv/s)^2) of the velocity change it makes (the dense
            solve's sparse jump cost), detections before the first and after the last flight cost a constant each.
The best chain gives positions and velocities per frame, a start for teacher_dense._teach.
"""
import os

import numpy as np

from birdseye_v2 import teacher_dense as TE

STRIDE = int(os.environ.get("T_HG_STRIDE", "3"))       # candidate hit frames every STRIDE frames (the solve refines)
DMIN, DMAX = int(os.environ.get("T_HG_DMIN", "8")), int(os.environ.get("T_HG_DMAX", "80"))  # frames between two hits
OFFS = (0.0,)                                           # m along the ray around its closest pass to a player
HG_PX = float(os.environ.get("T_HG_PX", "8.0"))         # px: pixel scale of the chain (its hit times and depths are on a coarse grid)
U_PX = 25.0                                             # px: a detection the chain leaves unexplained costs as much as this residual


def _obs(V, fr):
    """Per frame index: list of (camera, px) of the fixed cameras."""
    per = [[] for _ in fr]
    for j, cam in enumerate(V.cams):
        if isinstance(cam, TE.ObsCams):
            continue
        for f, p in zip(V.frames[j], V.px[j]):
            i = int(round(f - fr[0]))
            if 0 <= i < len(fr):
                per[i].append((j, np.asarray(p, float)))
    return per


def _obs_arrays(V, fr):
    """Per fixed camera: (T,2) detection px, NaN where none."""
    out = []
    for j, cam in enumerate(V.cams):
        if isinstance(cam, TE.ObsCams):
            continue
        A = np.full((len(fr), 2), np.nan)
        i = np.round(V.frames[j] - fr[0]).astype(int)
        m = (i >= 0) & (i < len(fr))
        A[i[m]] = V.px[j][m]
        out.append((cam, A))
    return out


def _path_cost(oa, F, X):
    """Robust pixel cost of paths: F (Q,d) frame indices, X (Q,d,3) positions -> (Q,)."""
    c = np.zeros(F.shape[0])
    for cam, A in oa:
        p = A[F]  # (Q,d,2)
        ok = np.isfinite(p[..., 0])
        if not ok.any():
            continue
        r = (cam.project(X[ok]) - p[ok]) / HG_PX
        cc = np.zeros(F.shape)
        cc[ok] = np.log1p(r * r).sum(1)
        c += cc.sum(1)
    return c


def _px_cost(V, per, i, X):
    """Robust pixel cost (dense solve's last stage) of frame i's detections against positions X (n,3)."""
    c = np.zeros(len(X))
    for j, p in per[i]:
        r = (V.cams[j].project(X) - p) / HG_PX
        c += np.log1p(r * r).sum(1)
    return c


def _nodes(V, fr, per, feet):
    nodes = []  # (frame index, X, kind) kind 0 hit, 1 landing
    for i in range(len(fr)):
        if not per[i]:
            continue
        for j, p in per[i]:
            c, d = V.cams[j].rays(p[None])
            c = np.asarray(c, float).reshape(-1)[:3]; d = d[0]
            if d[2] < -1e-3:
                s = -c[2] / d[2]
                X = c + s * d
                if -3 < X[0] < 9.1 and -3 < X[1] < 16.4:
                    nodes.append((i, X, 1))
            if i % STRIDE:
                continue
            dxy = d[:2]; n2 = dxy @ dxy
            if n2 < 1e-6:
                continue
            for F in feet[i]:
                if not np.isfinite(F).all():
                    continue
                s0 = (F - c[:2]) @ dxy / n2
                for o in OFFS:
                    s = s0 + o / np.sqrt(n2)
                    X = c + s * d
                    if s > 0.5 and np.linalg.norm(X[:2] - F) < TE.REACH and 0.2 < X[2] < TE.MAX_HIT_Z:
                        if not any(n[0] == i and n[2] == 0 and np.linalg.norm(n[1] - X) < 0.5 for n in nodes[-12:]):
                            nodes.append((i, X, 0))
    return nodes


def _step1(P, W, dt, k):
    """One RK4 step of gravity + drag (the dense solve refines with sub-steps)."""
    acc = lambda v: TE.G - k * np.linalg.norm(v, axis=-1, keepdims=True) * v
    a1 = acc(W); a2 = acc(W + 0.5 * dt * a1); a3 = acc(W + 0.5 * dt * a2); a4 = acc(W + dt * a3)
    P = P + dt * W + dt * dt / 6 * (a1 + a2 + a3)
    return P, W + dt / 6 * (a1 + 2 * a2 + 2 * a3 + a4)


def _shoot(Xa, Xb, n, dt, k, iters=7):
    """Initial velocities of drag flights from Xa to Xb in n frames (all rows the same n) -> v0, ok, path (n+1,m,3), vel."""
    T = n * dt
    v = (Xb - Xa) / T - 0.5 * TE.G * T
    def fly(v0):
        P, W = Xa.copy(), v0.copy()
        for _ in range(n):
            P, W = _step1(P, W, dt, k)
        return P
    eps = 0.01
    for _ in range(iters):
        E = fly(v) - Xb
        J = np.empty((len(v), 3, 3))
        for a in range(3):
            dv = np.zeros(3); dv[a] = eps
            J[:, :, a] = (fly(v + dv) - Xb - E) / eps
        try:
            st = np.linalg.solve(J, -E[..., None])[..., 0]
        except np.linalg.LinAlgError:
            st = -E / T
        nrm = np.linalg.norm(st, axis=1, keepdims=True)
        v = v + st * np.minimum(1.0, 15.0 / np.maximum(nrm, 1e-9))
    path, vel = [Xa.copy()], [v.copy()]
    P, W = Xa.copy(), v.copy()
    for _ in range(n):
        P, W = _step1(P, W, dt, k)
        path.append(P); vel.append(W)
    path, vel = np.array(path), np.array(vel)
    ok = (np.linalg.norm(path[-1] - Xb, axis=1) < 0.05) & (np.linalg.norm(v, axis=1) < 110) \
        & (path[..., 2].min(0) > -0.05) & (path[..., 2].max(0) < 12)
    return v, ok, path, vel


def chain(V, fr, k, sig_dv):
    """-> (P (T,3), Vv (T,3), covered (T,) bool, hits [frame...]) or None."""
    dt = 1.0 / V.fps
    per = _obs(V, fr)
    oa = _obs_arrays(V, fr)
    feet = TE._players(V, fr)
    nodes = _nodes(V, fr, per, feet)
    if len(nodes) < 2:
        return None
    ti = np.array([n[0] for n in nodes]); X = np.array([n[1] for n in nodes]); kind = np.array([n[2] for n in nodes])
    nobs = np.array([len(o) for o in per])
    cum = np.r_[0, np.cumsum(nobs)]
    U = np.log1p((U_PX / HG_PX) ** 2) * 2
    # edges, grouped by length
    E = []  # (a, b, v0, vend, cost, path, vel)
    for d in range(DMIN, DMAX + 1):
        a_idx, b_idx = [], []
        for a in np.nonzero(kind == 0)[0]:
            bs = np.nonzero(ti == ti[a] + d)[0]
            if len(bs):
                bs = bs[np.linalg.norm(X[bs] - X[a], axis=1) < 40 * d * dt]
                a_idx += [a] * len(bs); b_idx += bs.tolist()
        if not a_idx:
            continue
        a_idx, b_idx = np.array(a_idx), np.array(b_idx)
        v0, ok, path, vel = _shoot(X[a_idx], X[b_idx], d, dt, k)
        if not ok.any():
            continue
        q = np.nonzero(ok)[0]
        F = ti[a_idx[q]][:, None] + np.arange(1, d + 1)[None]
        cst = _path_cost(oa, F, path[1:, q].transpose(1, 0, 2))
        for n_, qq in enumerate(q):
            E.append((a_idx[qq], b_idx[qq], v0[qq], vel[-1, qq], cst[n_], path[:, qq], vel[:, qq]))
    if not E:
        return None
    ea = np.array([e[0] for e in E]); eb = np.array([e[1] for e in E])
    ev0 = np.array([e[2] for e in E]); ev1 = np.array([e[3] for e in E]); ec = np.array([e[4] for e in E])
    hitc = lambda dv: np.log1p((np.linalg.norm(dv, axis=-1) / sig_dv) ** 2)
    # start: everything before the first hit unexplained, the serve from rest
    best = U * cum[ti[ea]] + np.array([_px_cost(V, per, ti[a], X[a][None])[0] if per[ti[a]] else 0 for a in ea]) \
        + hitc(ev0) + ec
    back = np.full(len(E), -1)
    order = np.argsort(ti[ea], kind="stable")
    into = {}
    for e in range(len(E)):
        into.setdefault(eb[e], []).append(e)
    for e2 in order:  # edges in order of start frame: all edges into its start node are final already
        ins = into.get(ea[e2], [])
        if not ins:
            continue
        ins = np.array(ins)
        c = best[ins] + hitc(ev0[e2] - ev1[ins]) + ec[e2]
        m = np.argmin(c)
        if c[m] < best[e2]:
            best[e2], back[e2] = c[m], ins[m]
    tot = best + U * (cum[-1] - cum[ti[eb] + 1])
    e = int(np.argmin(tot))
    path = []
    while e >= 0:
        path.append(e); e = back[e]
    path = path[::-1]
    T = len(fr)
    P = np.full((T, 3), np.nan); Vv = np.full((T, 3), np.nan)
    hits = []
    for e in path:
        a, b = ea[e], eb[e]
        P[ti[a]:ti[b] + 1] = E[e][5]; Vv[ti[a]:ti[b] + 1] = E[e][6]
        hits.append(float(fr[ti[a]]))
    cov = np.isfinite(P[:, 0])
    return P, Vv, cov, hits, float(tot.min())
