"""Two-view physics teacher for the CCTV rallies (training-time only; the student never runs this).

The rally's shuttle is a chain of gravity + drag flights. Flights are found from the data alone: a stretch that one
flight cannot explain in BOTH cameras (pixel rms over the TV and side detections) hides a hit, and is split where
two flights explain it best; neighbours that one flight explains are merged back. The chain is then fitted jointly
to both views (robust pixel residuals), each flight starting where the previous one ended, never below the floor.

Out: dense 3D for every frame of the chain (gaps and stretches outside both frames included), the hit frames, and
each flight's support (detections per view) and rms per view -- the student's pseudo-labels and their confidence.
"""
import os

import numpy as np
from scipy.optimize import least_squares

from . import shuttle3d as S3

SIG_PX = 3.0      # px, soft-L1 scale of the reprojection residuals
SIG_CONT = 0.05   # m, consecutive flights meet at the hit
SIG_FLOOR = 0.02  # m
SPLIT_RMS = float(os.environ.get("SPLIT_RMS", "8.0"))  # px: a stretch one flight explains worse than this in either view hides a hit
MIN_OBS = 4       # detections (both views together) on each side of a split


class Views:
    """Detections of both cameras on the TV frame clock: cams[j], frames[j] (float), px[j] (n,2)."""

    def __init__(self, cams, frames, px, foot=None, C=None, rk_dir=None, rk_conf=None):
        self.cams = cams
        self.foot, self.C, self.rk_dir, self.rk_conf = foot, C, rk_dir, rk_conf  # TV-frame player context (rally_obs)
        self.fps = S3.FPS
        self.frames = [np.asarray(f, float) for f in frames]
        self.px = [np.asarray(p, float).reshape(-1, 2) for p in px]

    def window(self, a, b):
        return [((f >= a) & (f <= b)) for f in self.frames]

    def count(self, a, b):
        return sum(int(m.sum()) for m in self.window(a, b))


def _robust(r):
    return np.sign(r) * np.sqrt(2 * (np.sqrt(1 + r * r) - 1))


def _init_flight(V, a, b, tri=None):
    """Launch state from the triangulated points in the window (least squares on a short ballistic arc), else
    from the TV rays at a guessed height."""
    fps = S3.FPS
    if tri is not None and len(tri):
        m = (tri[:, 0] >= a) & (tri[:, 0] <= b)
        if m.sum() >= 3:
            t = (tri[m, 0] - a) / fps
            A = np.c_[np.ones_like(t), t]
            g = np.array([0.0, 0.0, -9.81])
            Y = tri[m, 1:] - 0.5 * np.outer(t ** 2, g)
            sol = np.linalg.lstsq(A, Y, rcond=None)[0]
            return np.r_[sol[0], sol[1]]
    mv = V.window(a, b)
    pts = [S3._ray_point(V.cams[0], V.px[0][mv[0]][k], 1.5) for k in (0, -1)] if mv[0].sum() >= 2 else None
    if pts is None:
        return np.r_[3.05, 6.7, 1.5, 0.0, 0.0, 0.0]
    T = max((b - a) / fps, 0.1)
    v = (pts[1] - pts[0]) / T
    v[2] += 0.5 * 9.81 * T
    return np.r_[pts[0], v]


def _flight_res(V, x, a, b):
    out = []
    for j, m in enumerate(V.window(a, b)):
        if m.any():
            X = S3.simulate(x[:3], x[3:6], (V.frames[j][m] - a) / S3.FPS)
            out.append(_robust(((V.cams[j].project(X) - V.px[j][m]) / SIG_PX).ravel()))
    return np.concatenate(out) if out else np.zeros(1)


def single_flight(V, a, b, tri=None, x0=None, nfev=200):
    """One flight over [a, b] fitted to both views: (params, rms pixel error per view). x0: warm start."""
    x0 = _init_flight(V, a, b, tri) if x0 is None else x0
    r = least_squares(lambda x: np.r_[_flight_res(V, x, a, b), np.minimum(S3.simulate(x[:3], x[3:6], np.linspace(0, (b - a) / S3.FPS, 8))[:, 2], 0) / SIG_FLOOR],
                      x0, max_nfev=nfev)
    rms = []
    for j, m in enumerate(V.window(a, b)):
        if m.sum():
            X = S3.simulate(r.x[:3], r.x[3:6], (V.frames[j][m] - a) / S3.FPS)
            e = np.linalg.norm(V.cams[j].project(X) - V.px[j][m], axis=1)
            rms.append(float(np.sqrt(np.mean(e ** 2))))
        else:
            rms.append(np.nan)
    return r.x, rms


def _bad(rms):
    return any(np.isfinite(e) and e > SPLIT_RMS for e in rms)


def segment(V, a, b, tri=None, step=2.0, max_depth=12):
    """Recursive residual-driven split of [a, b] into flights one flight each explains in both views."""
    _, rms = single_flight(V, a, b, tri)
    if not _bad(rms) or V.count(a, b) < 2 * MIN_OBS or max_depth == 0:
        return [(a, b)]
    allf = np.sort(np.concatenate([f[(f >= a) & (f <= b)] for f in V.frames]))
    best = None
    for tau in np.arange(allf[MIN_OBS - 1] + 0.5, allf[-MIN_OBS] - 0.5 + 1e-9, step):
        if V.count(a, tau) < MIN_OBS or V.count(tau, b) < MIN_OBS:
            continue
        e = 0.0
        for lo, hi in ((a, tau), (tau, b)):
            _, r = single_flight(V, lo, hi, tri)
            n = [int(m.sum()) for m in V.window(lo, hi)]
            e += sum(x * x * k for x, k in zip(r, n) if np.isfinite(x))
        if best is None or e < best[0]:
            best = (e, tau)
    if best is None:
        return [(a, b)]
    tau = best[1]
    for d in (-1.0, -0.5, 0.5, 1.0):  # refine the split to half a frame
        t2 = tau + d
        if V.count(a, t2) < MIN_OBS or V.count(t2, b) < MIN_OBS:
            continue
        e = 0.0
        for lo, hi in ((a, t2), (t2, b)):
            _, r = single_flight(V, lo, hi, tri)
            n = [int(m.sum()) for m in V.window(lo, hi)]
            e += sum(x * x * k for x, k in zip(r, n) if np.isfinite(x))
        if e < best[0]:
            best = (e, t2)
    tau = best[1]
    return segment(V, a, tau, tri, step, max_depth - 1) + segment(V, tau, b, tri, step, max_depth - 1)


def segment_greedy(V, tri=None, grow=2):
    """Flights grown forward in time: extend the current flight a few detections at a time (warm-started fit) while
    one flight still explains both views; when it stops, the hit is in the last step -- placed at the time that
    best explains the two sides -- and the next flight starts there."""
    allf = np.unique(np.concatenate(V.frames))
    shots, i0 = [], 0
    while i0 < len(allf) - 1:
        a = allf[i0]
        j = min(i0 + 2 * MIN_OBS, len(allf) - 1)
        x, _ = single_flight(V, a, allf[j], tri)
        good_j, good_x = j, x
        split = None
        while j < len(allf) - 1:
            j2 = min(j + grow, len(allf) - 1)
            x2, rms = single_flight(V, a, allf[j2], tri, x0=good_x, nfev=60)
            if _bad(rms):
                x2, rms = single_flight(V, a, allf[j2], tri)  # a cold start before declaring a hit
            if _bad(rms):
                split = (good_j, j2)
                break
            good_j, good_x, j = j2, x2, j2
        if split is None:
            shots.append((a, allf[-1]))
            break
        # hit between allf[good_j] and allf[j2]: best time there (half-frame grid), judged on the window around it
        lo_i, hi_i = split
        lo, hi = allf[lo_i], allf[hi_i]
        right_end = allf[min(hi_i + 2 * MIN_OBS, len(allf) - 1)]
        best = None
        for tau in np.arange(max(lo - 2, a + 1), hi + 0.5, 0.5):
            if V.count(tau, right_end) < MIN_OBS:
                continue
            e = 0.0
            for (p, q), x0 in (((a, tau), good_x), ((tau, right_end), None)):
                _, r = single_flight(V, p, q, tri, x0=x0, nfev=60)
                n = [int(m.sum()) for m in V.window(p, q)]
                e += sum(v * v * k for v, k in zip(r, n) if np.isfinite(v))
            if best is None or e < best[0]:
                best = (e, tau)
        tau = best[1] if best else hi
        shots.append((a, tau))
        i0 = int(np.searchsorted(allf, tau))
        if i0 < len(allf) and allf[i0] < tau:
            i0 += 1
        if i0 >= len(allf) - MIN_OBS:  # too little left for a flight of its own
            shots[-1] = (a, allf[-1])
            break
        allf = np.r_[allf[:i0], allf[i0:]]
        if allf[i0] > tau:  # the next flight starts at the hit, not at its first detection
            allf = np.r_[allf[:i0], [tau], allf[i0:]]
    return shots


def merge(V, shots, tri=None):
    """Merge adjacent flights that one flight explains in both views (removes spurious hits)."""
    shots = list(shots)
    while len(shots) > 1:
        best = None
        for i in range(len(shots) - 1):
            _, rms = single_flight(V, shots[i][0], shots[i + 1][1], tri)
            if not _bad(rms):
                e = max(x for x in rms if np.isfinite(x))
                if best is None or e < best[0]:
                    best = (e, i)
        if best is None:
            break
        i = best[1]
        shots[i:i + 2] = [(shots[i][0], shots[i + 1][1])]
    return shots


def merge_same_way(V, shots, tri=None):
    """Every hit sends the shuttle back toward the other side: two consecutive flights travelling the same way along
    the court were never separated by a hit (a stray detection split them) -- merge them."""
    shots = list(shots)
    vy = [single_flight(V, a, b, tri)[0][4] for a, b in shots]
    i = 0
    while i < len(shots) - 1:
        if np.sign(vy[i]) == np.sign(vy[i + 1]):
            shots[i:i + 2] = [(shots[i][0], shots[i + 1][1])]
            vy[i:i + 2] = [single_flight(V, *shots[i], tri)[0][4]]
            i = max(i - 1, 0)
        else:
            i += 1
    return shots


def fit_chain(V, shots, tri=None):
    """All flights jointly: both views' pixels, continuity at the hits, floor."""
    n = len(shots)
    x0 = np.concatenate([single_flight(V, a, b, tri)[0] for a, b in shots])

    def res(x):
        out = []
        for i, (a, b) in enumerate(shots):
            xi = x[6 * i:6 * i + 6]
            out.append(_flight_res(V, xi, a, b))
            ts = np.r_[(b - a) / S3.FPS, np.linspace(0, (b - a) / S3.FPS, 8)]
            X = S3.simulate(xi[:3], xi[3:6], ts)
            out.append(np.minimum(X[1:, 2], 0) / SIG_FLOOR)
            if i + 1 < n:
                out.append((X[0] - x[6 * (i + 1):6 * (i + 1) + 3]) / SIG_CONT)
        return np.concatenate(out)
    r = least_squares(res, x0, x_scale="jac", max_nfev=400)
    return r.x


def teach(V, tri=None):
    """-> dict(shots, params, hits, support (per flight: detections per view), rms (per flight per view)).
    hits: the flight boundaries where the shuttle turns back along the court (every real hit does; a boundary
    without a turn is a split by stray detections, kept for the positions but not reported as a hit)."""
    allf = np.concatenate(V.frames)
    a, b = float(allf.min()), float(allf.max())
    shots = merge(V, segment_greedy(V, tri), tri)
    x = fit_chain(V, shots, tri)
    support, rms = [], []
    for i, (lo, hi) in enumerate(shots):
        xi = x[6 * i:6 * i + 6]
        support.append([int(m.sum()) for m in V.window(lo, hi)])
        e = []
        for j, m in enumerate(V.window(lo, hi)):
            if m.any():
                X = S3.simulate(xi[:3], xi[3:6], (V.frames[j][m] - lo) / S3.FPS)
                e.append(round(float(np.sqrt(np.mean(np.sum((V.cams[j].project(X) - V.px[j][m]) ** 2, 1)))), 2))
            else:
                e.append(None)
        rms.append(e)
    return {"shots": [[float(lo), float(hi)] for lo, hi in shots], "params": x.tolist(),
            "hits": [float(shots[i][0]) for i in range(1, len(shots)) if x[6 * i + 4] * x[6 * (i - 1) + 4] < 0],
            "support": support, "rms": rms}


def positions(out, t):
    """3D at TV frame times t (NaN outside the chain)."""
    t = np.asarray(t, float)
    P = np.full((len(t), 3), np.nan)
    x = np.asarray(out["params"])
    for i, (lo, hi) in enumerate(out["shots"]):
        m = (t >= lo) & (t <= hi)
        if m.any():
            P[m] = S3.simulate(x[6 * i:6 * i + 3], x[6 * i + 3:6 * i + 6], (t[m] - lo) / S3.FPS)
    return P
