"""Two-view (top + back drone) time sync and shuttle triangulation for DroneData ground truth.

The two drones are not genlocked, so the back-view clock is  t_back = t_top + offset  with a
fractional-frame offset. A shuttle moves up to ~1 m per frame, so the offset is estimated to
0.05 frame by minimising the 3D ray-to-ray distance of the two views' 2D shuttle tracks.
"""
import csv

import numpy as np
from scipy.optimize import least_squares


def load_track(path):
    """TrackNet csv (Frame,Visibility,X,Y) or relabeled csv -> (N,) frames, (N,2) px  (visible only)."""
    fr, px = [], []
    for r in csv.DictReader(open(path)):
        if int(float(r.get("Visibility", 1))) and float(r["X"]) > 0:
            fr.append(int(r["Frame"]))
            px.append((float(r["X"]), float(r["Y"])))
    return np.array(fr, int), np.array(px, float).reshape(-1, 2)


def interp_track(frames, px, t, max_gap=1):
    """Linear interpolation of a 2D track at fractional times t; NaN where either neighbour is
    missing or more than max_gap frames apart (no bridging across occlusions)."""
    out = np.full((len(t), 2), np.nan)
    if len(frames) < 2:
        return out
    idx = np.searchsorted(frames, t, side="right") - 1
    ok = (idx >= 0) & (idx < len(frames) - 1)
    i = idx[ok]
    f0, f1 = frames[i], frames[i + 1]
    good = (f1 - f0) <= max_gap
    a = ((t[ok] - f0) / np.maximum(f1 - f0, 1))[:, None]
    val = px[i] * (1 - a) + px[i + 1] * a
    exact = np.isin(t, frames)  # exact integer hits always allowed
    res = np.where(good[:, None], val, np.nan)
    out[ok] = res
    if exact.any():
        lut = dict(zip(frames.tolist(), px))
        for k in np.where(exact)[0]:
            out[k] = lut[int(t[k])]
    return out


def ray_distance(c1, d1, c2, d2):
    """Closest-approach distance between ray sets (broadcast) + midpoints."""
    w = c1 - c2
    a = (d1 * d1).sum(-1); b = (d1 * d2).sum(-1); c = (d2 * d2).sum(-1)
    d = (d1 * w).sum(-1); e = (d2 * w).sum(-1)
    den = a * c - b * b
    s = (b * e - c * d) / den
    u = (a * e - b * d) / den
    p1 = c1 + s[..., None] * d1
    p2 = c2 + u[..., None] * d2
    return np.linalg.norm(p1 - p2, axis=-1), (p1 + p2) / 2


def estimate_offset(top_cam, back_cam, top_tr, back_tr, search=(-30, 30), cap=0.5, min_overlap=0.7):
    """Grid search (1 frame) then refine (0.05 frame) the back-minus-top time offset.
    Score = mean of min(ray distance, cap) over matched samples. Offsets whose overlap is below
    min_overlap x the best achievable overlap are rejected: with a handful of matches some
    far-off offset always finds a spuriously consistent pair set.
    Note the back camera looks down the court length, so shuttle motion along the length
    runs along its epipolar lines and barely constrains the offset; lateral motion and hits do."""
    tf, tp = top_tr
    bf, bp = back_tr
    if len(tf) < 10 or len(bf) < 10:
        return None
    ct, dt = top_cam.rays(tp)

    def score(off):
        q = interp_track(bf, bp, tf + off)
        ok = ~np.isnan(q[:, 0])
        if ok.sum() < 5:
            return np.inf, int(ok.sum())
        cb, db = back_cam.rays(q[ok])
        dist, _ = ray_distance(ct, dt[ok], cb, db)
        return float(np.minimum(dist, cap).mean()), int(ok.sum())

    coarse = [(o,) + score(o) for o in range(search[0], search[1] + 1)]
    nmax = max(c[2] for c in coarse)
    valid = [c for c in coarse if c[2] >= min_overlap * nmax]
    best = min(valid, key=lambda c: c[1])[0]
    fine = [(o,) + score(o) for o in np.arange(best - 1, best + 1.0001, 0.05)]
    fine = [c for c in fine if c[2] >= min_overlap * nmax]
    off, s, n = min(fine, key=lambda c: c[1])
    far = [c[1] for c in valid if abs(c[0] - off) >= 3]
    return {"offset": float(off), "score_m": float(s), "n": int(n),
            "second_best_ratio": float(min(far) / max(s, 1e-6)) if far else None}


def triangulate_px(cams, uvs):
    """Least-squares 3D point from >=2 views, minimising pixel reprojection error."""
    c0, d0 = cams[0].rays(uvs[0][None])
    c1, d1 = cams[1].rays(uvs[1][None])
    _, X0 = ray_distance(c0, d0, c1, d1)

    def res(X):
        return np.concatenate([cam.project(X[None])[0] - uv for cam, uv in zip(cams, uvs)])

    r = least_squares(res, X0[0], method="lm")
    return r.x, np.abs(r.fun).reshape(-1, 2)


def triangulate_track(top_cam, back_cam, top_tr, back_tr, offset, max_gap_m=0.2, max_err_px=None):
    """3D shuttle points at every top-view frame that has a back-view match at t+offset.

    Gate = closest-approach distance of the two viewing rays (metres). Pixel reprojection error
    is a poor gate here: ~30 px is only ~12 cm at these ranges, and the painted court is not
    exactly regulation, so calibration alone leaves ~5-15 cm of two-view inconsistency.
    Returns list of dicts {frame (top clock), X, gap_m, err_top_px, err_back_px}."""
    tf, tp = top_tr
    bf, bp = back_tr
    q = interp_track(bf, bp, tf + offset)
    out = []
    for f, uvt, uvb in zip(tf, tp, q):
        if np.isnan(uvb[0]):
            continue
        c0, d0 = top_cam.rays(uvt[None])
        c1, d1 = back_cam.rays(uvb[None])
        gap, _ = ray_distance(c0, d0, c1, d1)
        if gap[0] > max_gap_m:
            continue
        X, err = triangulate_px([top_cam, back_cam], [uvt, uvb])
        if max_err_px is not None and err.max() > max_err_px:
            continue
        if -0.3 < X[2] < 15:
            out.append({"frame": int(f), "X": X.tolist(), "gap_m": float(gap[0]),
                        "err_top_px": float(np.hypot(*err[0])), "err_back_px": float(np.hypot(*err[1]))})
    return out


def kink_time(frames, px, t_guess, win=6, step=0.05):
    """Sub-frame time of a hit (velocity discontinuity) in a 2D track near t_guess.
    Fits a continuous piecewise-quadratic in time with a knot at tau (independently per image
    axis) and returns the tau with the lowest residual, or None if either side has <3 samples."""
    m = np.abs(frames - t_guess) <= win
    t, p = frames[m].astype(float), px[m]
    best = None
    for tau in np.arange(t_guess - win + 2, t_guess + win - 2 + 1e-9, step):
        d = t - tau
        if (d < 0).sum() < 3 or (d > 0).sum() < 3:
            continue
        A = np.stack([np.ones_like(d), d, d ** 2, np.maximum(d, 0), np.maximum(d, 0) ** 2], 1)
        sse = sum(np.linalg.lstsq(A, p[:, k], rcond=None)[1].sum() for k in range(2))
        if best is None or sse < best[0]:
            best = (sse, tau)
    return None if best is None else float(best[1])


def offset_from_hits(top_tr, back_tr, hit_frames, guess=0.0, search=4):
    """Back-minus-top offset from hit instants seen in both views (independent of calibration).
    hit_frames: annotated hit frames in the top-view clock. Returns (median offset, per-hit list)."""
    per = []
    for h in hit_frames:
        tt = kink_time(top_tr[0], top_tr[1], h, win=6)
        if tt is None:
            continue
        tb = kink_time(back_tr[0], back_tr[1], tt + guess, win=6 + search)
        if tb is not None:
            per.append(tb - tt)
    return (float(np.median(per)) if per else None), per
