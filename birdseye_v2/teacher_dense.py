"""Dense two-view physics teacher for the CCTV rallies (training-time only; the student never runs this).

The rally's shuttle is one state (position, velocity) per TV frame. Between consecutive frames the state follows
gravity + quadratic drag exactly (RK4 sub-steps); the shuttle's position is continuous, its velocity may jump -- a
hit -- at an L1 cost, so jumps are few. A jump costs less where a player can reach (within REACH of a player's feet
on the court, below MAX_HIT_Z): a TV-only flight is thereby pinned in depth between the two players who hit it.
Detections of both cameras enter as robust (Cauchy) pixel residuals; the floor is a soft wall.
No segmentation search: one sparse least-squares problem per rally.

Confidence: the Laplace approximation of the fit (inverse Gauss-Newton Hessian) gives each frame's 3D standard
deviation; frames above CONF_SIGMA are not labels (the student does not learn from what the teacher cannot see).
Hits: where the velocity along the court reverses, in the stretches the teacher is sure of (hits_of).

Interface as birdseye_v2.teacher: teach(V, tri) -> dict, positions(out, t), confidence(out, t).
"""
import os

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

G = np.array([0.0, 0.0, -9.81])
V_TERM = 6.4
SIG_PX = float(os.environ.get("T_SIG_PX", "3.0"))        # px, Cauchy scale of the reprojection residuals
SIG_POS = float(os.environ.get("T_SIG_POS", "0.002"))    # m, position continuity frame to frame (tight: slack here is a hidden impulse)
SIG_DV = float(os.environ.get("T_SIG_DV", "3.0"))        # m/s, scale of the L1 velocity jumps
AIR = float(os.environ.get("T_AIR", "30.0"))             # extra cost factor of a jump out of the players' reach
REACH = float(os.environ.get("T_REACH", "2.0"))          # m, horizontal reach around a player's feet
# measured hit zone (129 two-view hits, CCTV sf): the shuttle is struck ~1.06 m in front of the hitter's feet (toward the
# net) and ~0.66 m to the -x side; spread ~0.7 m. T_HITZONE=1: jumps are cheap within HZ_R of that spot (Mahalanobis)
HITZONE = os.environ.get("T_HITZONE", "0") == "1"
HZ_C = (-0.66, 1.06)   # m: (across x, toward net)
HZ_S = (0.75, 0.70)    # m: spread
HZ_R = float(os.environ.get("T_HZ_R", "2.0"))
MAX_HIT_Z = float(os.environ.get("T_MAX_HIT_Z", "3.6"))  # m
SIG_FLOOR = 0.02
SUB = 6                                                  # RK4 sub-steps per frame
CONF_SIGMA = float(os.environ.get("T_CONF", "0.15"))     # m
HIT_SD = float(os.environ.get("T_HIT_SD", "0.3"))        # m, the teacher labels hits (and their absence) below this
# T_DV_SPARSE=<m/s>: a last stage where a velocity jump costs log(1 + (dv/s)^2) -- about the same for any real hit, so
# a hit is an event (racket), never a stream of small pushes (which the L1 cost allows: a "hovering" shuttle)
DV_SPARSE = float(os.environ.get("T_DV_SPARSE", "0"))


def _cauchy(r):
    return np.sqrt(np.log1p(r * r))


def _cauchy_vec(r):
    """Cauchy on the length of each row's 3-vector, spread over its 3 entries (a jump's cost does not depend on its
    direction)."""
    n = np.linalg.norm(r, axis=-1, keepdims=True)
    return r / np.maximum(n, 1e-9) * np.sqrt(np.log1p(n * n))


def _l1(r):
    return np.sign(r) * np.sqrt(2 * (np.sqrt(1 + r * r) - 1))


def step(P, Vv, dt, k=9.81 / V_TERM ** 2):
    """Exact-ish (RK4, SUB sub-steps) gravity + drag transition of many states at once."""
    h = dt / SUB
    acc = lambda v: G - k * np.linalg.norm(v, axis=-1, keepdims=True) * v
    for _ in range(SUB):
        k1v = acc(Vv); k1p = Vv
        k2v = acc(Vv + 0.5 * h * k1v); k2p = Vv + 0.5 * h * k1v
        k3v = acc(Vv + 0.5 * h * k2v); k3p = Vv + 0.5 * h * k2v
        k4v = acc(Vv + h * k3v); k4p = Vv + h * k3v
        P = P + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        Vv = Vv + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return P, Vv


class ObsCams:
    """A panning / zooming camera seen as one pose per observation: R (m,3,3) world->camera, focal f (m,), fixed centre
    C, radial k1, principal point at the image centre. Behaves like a Camera for project / rays over its observations
    (row i of X is projected with pose i); subset(mask) keeps the poses of the kept observations."""

    PX_SCALE = float(os.environ.get("T_PCAM_PX", "3.0"))  # x SIG_PX: per-frame calibration and sync are looser than the fixed cameras'

    def __init__(self, R, f, C, k1, size=(864, 486)):
        self.R, self.f = np.asarray(R, float).reshape(-1, 3, 3), np.asarray(f, float).ravel()
        self.C, self.k1, self.size = np.asarray(C, float), float(k1), size

    def subset(self, m):
        return ObsCams(self.R[m], self.f[m], self.C, self.k1, self.size)

    def project(self, X):
        Y = np.einsum("mij,mj->mi", self.R, np.asarray(X, float).reshape(-1, 3) - self.C)
        z = np.maximum(Y[:, 2], 1e-3)
        x, y = Y[:, 0] / z, Y[:, 1] / z
        d = 1 + self.k1 * (x * x + y * y)
        return np.c_[self.f * x * d + self.size[0] / 2, self.f * y * d + self.size[1] / 2]

    def rays(self, uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        x, y = (uv[:, 0] - self.size[0] / 2) / self.f, (uv[:, 1] - self.size[1] / 2) / self.f
        xu, yu = x.copy(), y.copy()
        for _ in range(5):  # undo the radial term
            d = 1 + self.k1 * (xu * xu + yu * yu)
            xu, yu = x / d, y / d
        dc = np.c_[xu, yu, np.ones(len(uv))]
        dw = np.einsum("mji,mj->mi", self.R, dc)
        return self.C, dw / np.linalg.norm(dw, axis=1, keepdims=True)


def _cam(cam, m):
    return cam.subset(m) if hasattr(cam, "subset") else cam


def _players(V, fr):
    """(T,4,2) player feet on the court at TV frames fr (NaN where unknown)."""
    T = len(fr)
    if V.foot is None:
        return np.full((T, 4, 2), np.nan)
    F = np.full((T, 4, 2), np.nan)
    idx = fr.astype(int)
    ok = (idx >= 0) & (idx < len(V.foot))
    F[ok] = V.foot[idx[ok]]
    return F


def _init(V, fr, tri):
    """Initial positions: triangulated points; else a ray point at the depth interpolated (in time) from the
    triangulated points' depth along that camera; else linear in time; velocity by differences."""
    T = len(fr)
    P = np.full((T, 3), np.nan)
    lo = fr[0]
    if tri is not None and len(tri):
        m = (tri[:, 0] >= fr[0]) & (tri[:, 0] <= fr[-1])
        P[(tri[m, 0] - lo).astype(int)] = tri[m, 1:]
    have = np.isfinite(P[:, 0])
    for j in range(len(V.cams)):
        f, px = V.frames[j], V.px[j]
        m = (f >= fr[0]) & (f <= fr[-1])
        if not m.any():
            continue
        c, d = _cam(V.cams[j], m).rays(px[m])
        if have.sum() >= 2:
            dep = np.linalg.norm(P[have] - c, axis=1)
            s = np.interp(f[m], fr[have], dep)
        else:
            s = np.full(m.sum(), np.linalg.norm(np.array([3.05, 6.7, 1.5]) - c))
        idx = (f[m] - lo).astype(int)
        new = ~np.isfinite(P[idx, 0])
        P[idx[new]] = c + s[new, None] * d[new]
        have = np.isfinite(P[:, 0])
    for a in range(3):
        P[~have, a] = np.interp(fr[~have], fr[have], P[have, a])
    P[:, 2] = np.maximum(P[:, 2], 0.05)
    Vv = np.gradient(P, axis=0) * V.fps
    return P, np.clip(Vv, -60, 60)


class Problem:
    def __init__(self, V, fr, k=9.81 / V_TERM ** 2):
        self.V, self.fr, self.T, self.k = V, fr, len(fr), k
        self.px_loss, self.sig_px = _l1, SIG_PX
        self.dv_loss, self.sig_dv = _l1, SIG_DV
        self.dt = 1.0 / V.fps
        self.feet = _players(V, fr)
        self.obs, self.cams, self.frac = [], [], []
        for j in range(len(V.cams)):
            f = V.frames[j]
            m = (f >= fr[0]) & (f <= fr[-1])
            # fractional times (player cameras are not frame-locked to the TV): position interpolated between frames
            u = f[m] - fr[0]
            idx = np.minimum(np.floor(u).astype(int), self.T - 1)
            a = np.where(idx < self.T - 1, u - idx, 0.0)
            self.obs.append((idx, V.px[j][m]))
            self.frac.append(a)
            self.cams.append(_cam(V.cams[j], m))

    def at(self, P, j):
        idx, a = self.obs[j][0], self.frac[j]
        if not a.any():
            return P[idx]
        return P[idx] * (1 - a)[:, None] + P[np.minimum(idx + 1, self.T - 1)] * a[:, None]

    def unpack(self, x):
        X = x.reshape(self.T, 6)
        return X[:, :3], X[:, 3:]

    def reach_weight(self, P, feet):
        """1 within a player's reach, 1 + AIR beyond (smooth); also above MAX_HIT_Z."""
        if HITZONE:
            toward = np.sign(6.7 - feet[..., 1])  # +1 for a player on the y < net side
            ex = (P[:, None, 0] - feet[..., 0] - HZ_C[0]) / HZ_S[0]
            ey = ((P[:, None, 1] - feet[..., 1]) * toward - HZ_C[1]) / HZ_S[1]
            d = np.hypot(ex, ey) * (REACH / HZ_R)  # in "REACH" units, so the sigmoid below is unchanged
        else:
            d = np.linalg.norm(feet - P[:, None, :2], axis=2)
        d = np.where(np.isfinite(d), d, 99.0).min(1)
        if not np.isfinite(feet).any():
            d = np.zeros(len(P))
        far = 1 / (1 + np.exp(-(d - REACH) / 0.15)) + 1 / (1 + np.exp(-(P[:, 2] - MAX_HIT_Z) / 0.15))
        return 1 + AIR * np.minimum(far, 1.0)

    def residuals(self, x):
        P, Vv = self.unpack(x)
        out = []
        for j, (idx, px) in enumerate(self.obs):
            sc = getattr(self.cams[j], "PX_SCALE", 1.0)
            out.append(self.px_loss(((self.cams[j].project(self.at(P, j)) - px) / (self.sig_px * sc))).ravel())
        P1, V1 = step(P[:-1], Vv[:-1], self.dt, self.k)
        out.append(((P[1:] - P1) / SIG_POS).ravel())
        w = self.reach_weight(P[1:], self.feet[1:])
        out.append((np.sqrt(w)[:, None] * self.dv_loss((Vv[1:] - V1) / self.sig_dv)).ravel())
        out.append(np.minimum(P[:, 2], 0) / SIG_FLOOR)
        return np.concatenate(out)

    def sparsity(self):
        T = self.T
        rows = sum(2 * len(idx) for idx, _ in self.obs) + 6 * (T - 1) + T
        S = lil_matrix((rows, 6 * T), dtype=int)
        r = 0
        for (idx, _), a in zip(self.obs, self.frac):
            for t, aa in zip(idx, a):
                S[r:r + 2, 6 * t:6 * t + 3] = 1
                if aa > 0:
                    S[r:r + 2, 6 * (t + 1):6 * (t + 1) + 3] = 1
                r += 2
        for t in range(T - 1):
            S[r:r + 3, 6 * t:6 * t + 12] = 1
            r += 3
        for t in range(T - 1):
            S[r:r + 3, 6 * t:6 * t + 12] = 1
            r += 3
        for t in range(T):
            S[r, 6 * t + 2] = 1
            r += 1
        return S


def teach(V, tri=None):
    """T_VTERM_GRID="5.8,6.4,7.0": the shuttle's terminal velocity (drag) is fitted per rally -- the solve is run for
    each value and the lowest-cost one kept (feathers and air vary; depth in one-camera flights comes from the curve)."""
    grid = os.environ.get("T_VTERM_GRID")
    if not grid:
        return _teach_global(V, tri, V_TERM)
    best = None
    for vt in [float(g) for g in grid.split(",")]:
        o = _teach_global(V, tri, vt)
        if best is None or o["cost"] < best["cost"]:
            best = o
    return best


def _teach_global(V, tri, v_term):
    """T_GLOBAL=1: also solved from the hit-graph chain (birdseye_v2.hitgraph); the lower final cost is kept (same
    objective, so the cost decides)."""
    o = _teach(V, tri, v_term)
    if os.environ.get("T_GLOBAL", "0") != "1":
        return o
    from birdseye_v2 import hitgraph
    allf = np.concatenate(V.frames)
    fr = np.arange(np.floor(allf.min()), np.ceil(allf.max()) + 1)
    ch = hitgraph.chain(V, fr, 9.81 / v_term ** 2, DV_SPARSE if DV_SPARSE > 0 else float(os.environ.get("T_HG_SIG", "0.5")))
    if ch is None:
        o["global"] = "none"
        return o
    P, Vv, cov, hits, _ = ch
    P0, V0 = _init(V, fr, tri)
    P0[cov], V0[cov] = P[cov], Vv[cov]
    g = _teach(V, tri, v_term, init=(P0, V0))
    g["global"] = "won" if g["cost"] < o["cost"] else "lost"
    g["cost_local"], g["cost_global"] = o["cost"], g["cost"]
    if g["cost"] < o["cost"]:
        return g
    o["global"], o["cost_global"] = "lost", g["cost"]
    return o


def _teach(V, tri, v_term, init=None):
    allf = np.concatenate(V.frames)
    fr = np.arange(np.floor(allf.min()), np.ceil(allf.max()) + 1)
    pb = Problem(V, fr, k=9.81 / v_term ** 2)
    P0, V0 = _init(V, fr, tri) if init is None else init
    x0 = np.c_[P0, V0].ravel()
    sc = np.tile([1, 1, 1, 10, 10, 10], len(fr)).astype(float)
    # graduated: L1 pixels (near-convex; every detection pulls) from a wide scale down, then Cauchy (outliers let go)
    S = pb.sparsity()
    x = x0
    stages = [(_l1, 4 * SIG_PX, _l1, SIG_DV), (_l1, SIG_PX, _l1, SIG_DV), (_cauchy, SIG_PX, _l1, SIG_DV)]
    if DV_SPARSE > 0:
        stages.append((_cauchy, SIG_PX, _cauchy_vec, DV_SPARSE))
    for loss, sp, dvl, sdv in stages:
        pb.px_loss, pb.sig_px, pb.dv_loss, pb.sig_dv = loss, sp, dvl, sdv
        r = least_squares(pb.residuals, x, jac_sparsity=S, x_scale=sc, method="trf", tr_solver="lsmr",
                          max_nfev=int(os.environ.get("T_NFEV", "150")))
        x = r.x
    P, Vv = pb.unpack(r.x)
    # Laplace: 3D standard deviation per frame from the Gauss-Newton Hessian
    J = r.jac.tocsc() if hasattr(r.jac, "tocsc") else r.jac
    sd_l1 = None
    if DV_SPARSE > 0 and os.environ.get("T_SD_L1", "0") in ("1", "both"):
        # the log jump cost is stiff around dv = 0 and makes the sd overconfident: the sd under the L1 jump cost at the
        # same solution ("1": used as the sd; "both": kept as sd_l1, an extra feature for the learned confidence)
        from scipy.optimize._numdiff import approx_derivative, group_columns
        pb.dv_loss, pb.sig_dv = _l1, SIG_DV
        S2 = S.tocsc()
        J2 = approx_derivative(pb.residuals, r.x, sparsity=(S2, group_columns(S2))).tocsc()
        if os.environ.get("T_SD_L1") == "1":
            J = J2
        else:
            H2 = (J2.T @ J2).toarray()
            try:
                sd_l1 = np.sqrt(np.clip(np.diag(np.linalg.inv(H2 + 1e-6 * np.eye(len(H2)))).reshape(-1, 6)[:, :3].sum(1), 0, None))
            except np.linalg.LinAlgError:
                sd_l1 = np.full(len(fr), np.inf)
    H = (J.T @ J)
    H = H.toarray() if hasattr(H, "toarray") else H
    try:
        Ci = np.linalg.inv(H + 1e-6 * np.eye(len(H)))
        sd = np.sqrt(np.clip(np.diag(Ci).reshape(-1, 6)[:, :3].sum(1), 0, None))
    except np.linalg.LinAlgError:
        sd = np.full(len(fr), np.inf)
    _, V1 = step(P[:-1], Vv[:-1], pb.dt, pb.k)
    dv = np.r_[0.0, np.linalg.norm(Vv[1:] - V1, axis=1)]
    hits, hit_mask = hits_of(fr, Vv, sd)
    rep = []
    for j, (idx, px) in enumerate(pb.obs):
        e = np.linalg.norm(pb.cams[j].project(pb.at(P, j)) - px, axis=1) if len(idx) else np.zeros(0)
        rep.append(round(float(np.median(e)), 2) if len(e) else None)
    return {"frames": fr.tolist(), "P": P.round(4).tolist(), "V": Vv.round(3).tolist(), "sd": sd.round(4).tolist(),
            "dv": dv.round(2).tolist(), "hits": hits, "hit_mask": hit_mask, "rep_med_px": rep, "cost": float(r.cost), "nfev": int(r.nfev),
            "v_term": v_term, **({"sd_l1": sd_l1.round(4).tolist()} if sd_l1 is not None else {}),
            "shots": [[float(fr[0]), float(fr[-1])]]}


def hits_of(fr, Vv, sd, max_sd=None):
    """Hits: every hit sends the shuttle back toward the other side, so a hit is where the velocity along the court
    changes sign (two-view hits: recall 0.97 within 2 frames). Only where the teacher is sure (sd < HIT_SD): there it
    can also say "no hit" -> hit_mask [[first, last], ...] frame ranges where the hit labels are complete."""
    max_sd = HIT_SD if max_sd is None else max_sd
    ok = np.asarray(sd) < max_sd
    vy = np.asarray(Vv)[:, 1]
    # a reversal, not a vertical fall flickering around zero: >= 1 m/s along the court within 3 frames on each side
    # (two-view hits: recall 0.97, 1.02 hits per labelled second; real play ~1/s)
    hits = [float(fr[i]) for i in range(1, len(fr)) if vy[i - 1] * vy[i] < 0 and ok[i] and ok[i - 1]
            and np.abs(vy[max(i - 3, 0):i]).max() > 1.0 and np.abs(vy[i:i + 3]).max() > 1.0]
    mask, i = [], 0
    while i < len(fr):
        if ok[i]:
            j = i
            while j + 1 < len(fr) and ok[j + 1]:
                j += 1
            mask.append([int(fr[i]), int(fr[j])])
            i = j + 1
        else:
            i += 1
    return hits, mask


def positions(out, t):
    fr = np.asarray(out["frames"])
    P = np.asarray(out["P"])
    t = np.asarray(t, float)
    res = np.full((len(t), 3), np.nan)
    m = (t >= fr[0]) & (t <= fr[-1])
    for a in range(3):
        res[m, a] = np.interp(t[m], fr, P[:, a])
    return res


def confidence(out, t, thr=None):
    thr = CONF_SIGMA if thr is None else thr
    fr = np.asarray(out["frames"])
    sd = np.asarray(out["sd"])
    t = np.asarray(t, float)
    m = (t >= fr[0]) & (t <= fr[-1])
    c = np.zeros(len(t), bool)
    c[m] = np.interp(t[m], fr, sd) < thr
    return c


# ---------------------------------------------------------------- exact chain (default teacher) -------------------
# Physics carries depth only when it is exact: between two hits the shuttle is in free flight, no impulse at all.
# The dense solve above only proposes hit candidates (its velocity jumps); the chain below is the model: flights of
# gravity + drag joined at hits with sub-frame times, each hit within a player's reach, hits the data does not need
# removed (Occam: a hit stays if removing it costs more than PRUNE).
from . import shuttle3d as S3

SIG_CONT = 0.01   # m, a flight ends where the next starts
SIG_REACH = 0.10  # m
PRUNE = float(os.environ.get("T_PRUNE", "25"))
CAND_DV = float(os.environ.get("T_CAND_DV", "2.0"))


class Chain:
    """Flights i = 0..n-1 start at tau[i] (tau[0] fixed at the first frame) with state (p_i, v_i); params: first
    flight [p(3), v(3)], each later one [tau, p(3), v(3)]."""

    def __init__(self, pb, t_first, t_last):
        self.pb, self.t0, self.t1 = pb, t_first, t_last
        self.vt = [pb.fr[0] + idx for idx, _ in pb.obs]  # detection times per view
        self.vpx = [px for _, px in pb.obs]
        self.det_t = np.concatenate(self.vt)

    @staticmethod
    def cols(i):
        return list(range(6)) if i == 0 else list(range(6 + 7 * (i - 1), 6 + 7 * i))

    def split(self, x, n):
        taus = np.r_[self.t0, x[6:6 + 7 * (n - 1)].reshape(-1, 7)[:, 0]] if n > 1 else np.array([self.t0])
        st = [x[:6]] + [x[6 + 7 * i + 1:6 + 7 * i + 7] for i in range(n - 1)]
        return taus, st

    def positions(self, x, n, t):
        taus, st = self.split(x, n)
        t = np.asarray(t, float)
        f = np.clip(np.searchsorted(taus, t, side="right") - 1, 0, n - 1)
        P = np.empty((len(t), 3))
        for i in np.unique(f):
            m = f == i
            P[m] = S3.simulate(st[i][:3], st[i][3:], (t[m] - taus[i]) / self.pb.V.fps)
        return P

    def residuals(self, x, n):
        pb = self.pb
        taus, st = self.split(x, n)
        out = []
        for j in range(len(self.vt)):
            if len(self.vt[j]):
                P = self.positions(x, n, self.vt[j])
                out.append(pb.px_loss((pb.V.cams[j].project(P) - self.vpx[j]) / pb.sig_px).ravel())
        feet = pb.feet
        for i in range(n):
            end = taus[i + 1] if i + 1 < n else self.t1
            T = max(end - taus[i], 0.5) / pb.V.fps
            X = S3.simulate(st[i][:3], st[i][3:], np.r_[np.linspace(0, T, 8), (end - taus[i]) / pb.V.fps])
            out.append(np.minimum(X[:-1, 2], 0) / SIG_FLOOR)
            if i + 1 < n:
                out.append((X[-1] - st[i + 1][:3]) / SIG_CONT)
            k = int(np.clip(round(taus[i] - pb.fr[0]), 0, len(feet) - 1))
            d = np.linalg.norm(feet[k] - st[i][:2], axis=1)
            d = np.nanmin(d) if np.isfinite(d).any() else 0.0
            out.append(np.array([max(d - REACH, 0) / SIG_REACH, max(st[i][2] - MAX_HIT_Z, 0) / SIG_REACH]))
        return np.concatenate(out)

    def sparsity(self, x, n):
        taus, _ = self.split(x, n)
        rows = []
        for j in range(len(self.vt)):
            for t in self.vt[j]:
                fs = set(np.clip(np.searchsorted(taus, [t - 1.5, t, t + 1.5], side="right") - 1, 0, n - 1))
                c = sum((self.cols(f) for f in fs), [])
                rows += [c, c]
        for i in range(n):
            rows += [self.cols(i)] * 8
            if i + 1 < n:
                rows += [self.cols(i) + self.cols(i + 1)] * 3
            rows += [self.cols(i)] * 2
        S = lil_matrix((len(rows), len(x)), dtype=int)
        for r, c in enumerate(rows):
            S[r, c] = 1
        return S

    def fit(self, x0, n, nfev=100, free=None):
        """free: flight indices whose params move (default all)."""
        taus, _ = self.split(x0, n)
        lo = np.full(len(x0), -np.inf)
        hi = np.full(len(x0), np.inf)
        for i in range(1, n):  # hit times move within +-1 frame of their start value
            lo[6 + 7 * (i - 1)] = taus[i] - 1.0
            hi[6 + 7 * (i - 1)] = taus[i] + 1.0
        sc = np.r_[[1, 1, 1, 10, 10, 10], np.tile([1, 1, 1, 1, 10, 10, 10], n - 1)].astype(float)
        idx = np.arange(len(x0)) if free is None else np.array(sum((self.cols(i) for i in free), []))
        S = self.sparsity(x0, n)[:, idx]
        xf = x0.copy()

        def res(z):
            xf[idx] = z
            return self.residuals(xf, n)
        r = least_squares(res, np.clip(x0[idx], lo[idx] + 1e-6, hi[idx] - 1e-6), bounds=(lo[idx], hi[idx]),
                          x_scale=sc[idx], jac_sparsity=S, method="trf", tr_solver="lsmr", max_nfev=nfev)
        x = x0.copy()
        x[idx] = r.x
        r.full_x = x
        return r


def _chain_x(P, Vv, fr, taus):
    idx = [int(np.clip(round(t - fr[0]), 0, len(fr) - 1)) for t in taus]
    x = [P[idx[0]], Vv[idx[0]]]
    for t, i in zip(taus[1:], idx[1:]):
        j = min(i + 1, len(fr) - 1)  # state just after the hit
        x.append(np.r_[t, P[i], Vv[j]])
    return np.concatenate(x)


def teach_chain(V, tri=None):
    dense = teach_dense(V, tri)
    fr = np.asarray(dense["frames"], float)
    P, Vv, dv = np.asarray(dense["P"]), np.asarray(dense["V"]), np.asarray(dense["dv"])
    pb = Problem(V, fr)
    pb.px_loss, pb.sig_px = _cauchy, SIG_PX
    dv2 = dv + np.r_[dv[1:], 0]
    cand = [i for i in range(2, len(fr) - 2) if dv2[i] > CAND_DV and dv2[i] >= dv2[i - 1] and dv2[i] >= dv2[i + 1]]
    keep = []
    for i in sorted(cand, key=lambda i: -dv2[i]):  # at least 3 frames apart
        if all(abs(i - k) >= 3 for k in keep):
            keep.append(i)
    taus = [fr[0]] + [fr[i] for i in sorted(keep)]
    ch = Chain(pb, fr[0], fr[-1])
    n = len(taus)
    r = ch.fit(_chain_x(P, Vv, fr, taus), n)
    x = r.full_x
    cost = r.cost
    # Occam: remove the weakest hits while the data does not need them
    tried = set()
    while n > 1:
        tz, st = ch.split(x, n)
        jump = [np.linalg.norm(st[i][3:] - (S3.simulate(st[i - 1][:3], st[i - 1][3:], [(tz[i] - tz[i - 1]) / V.fps, (tz[i] - tz[i - 1]) / V.fps + 1e-3])[1]
                                              - S3.simulate(st[i - 1][:3], st[i - 1][3:], [(tz[i] - tz[i - 1]) / V.fps])[0]) / 1e-3)
                for i in range(1, n)]
        order = [i for i in np.argsort(jump) + 1 if round(tz[i], 1) not in tried]
        if not order:
            break
        i = order[0]
        tried.add(round(tz[i], 1))
        x2 = np.r_[x[:6 + 7 * (i - 1)], x[6 + 7 * i:]]
        r2 = ch.fit(x2, n - 1, nfev=40, free=[k for k in range(i - 2, i + 1) if 0 <= k < n - 1])
        if r2.cost - cost < PRUNE:
            x, n, cost = r2.full_x, n - 1, r2.cost
            tried = set()
    r = ch.fit(x, n, nfev=200)
    r.x = r.full_x
    tz, st = ch.split(r.x, n)
    # Laplace: parameter covariance, propagated to each frame's position
    J = r.jac.toarray() if hasattr(r.jac, "toarray") else r.jac
    Cp = np.linalg.pinv(J.T @ J)
    Pf = ch.positions(r.x, n, fr)
    Jp = np.zeros((len(fr), 3, len(r.x)))
    for k in range(len(r.x)):
        if k >= 6 and (k - 6) % 7 == 0:
            continue  # hit time: frames switch flights discontinuously; its effect is in the neighbours' states
        e = np.zeros(len(r.x)); e[k] = 1e-4 * (10 if (k < 6 and k >= 3) or (k >= 6 and (k - 6) % 7 >= 4) else 1)
        Jp[:, :, k] = (ch.positions(r.x + e, n, fr) - Pf) / e[k]
    sd = np.sqrt(np.clip(np.einsum("tak,kl,tal->t", Jp, Cp, Jp), 0, None))
    jumps = []
    for i in range(1, n):
        a = S3.simulate(st[i - 1][:3], st[i - 1][3:], [(tz[i] - tz[i - 1]) / V.fps, (tz[i] - tz[i - 1]) / V.fps + 1e-3])
        jumps.append(float(np.linalg.norm(st[i][3:] - (a[1] - a[0]) / 1e-3)))
    rep = []
    for j, (idx, px) in enumerate(pb.obs):
        e = np.linalg.norm(V.cams[j].project(Pf[idx]) - px, axis=1) if len(idx) else np.zeros(0)
        rep.append(round(float(np.median(e)), 2) if len(e) else None)
    return {"frames": fr.tolist(), "P": Pf.round(4).tolist(), "sd": sd.round(4).tolist(),
            "hits": [float(t) for t in tz[1:]], "hit_dv": [round(j, 2) for j in jumps],
            "params": r.x.tolist(), "n_flights": n, "rep_med_px": rep, "cost": float(r.cost),
            "shots": [[float(tz[i]), float(tz[i + 1] if i + 1 < n else fr[-1])] for i in range(n)],
            "candidates": len(keep)}


teach_dense = teach
if os.environ.get("T_CHAIN") == "1":
    teach = teach_chain
