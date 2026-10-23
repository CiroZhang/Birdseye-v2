"""Physically plausible bodies: smooth ground path + foot-contact lock (stabilize()).

Per-frame monocular bodies (VIMO) jitter in ground position, so a foot that is standing still on the
floor slides around (DroneData test: median 1.6 m/s for feet < 5 cm off the floor). A foot is planted
while it is near the floor and slow (after light smoothing); over each planted stretch the foot is
pinned to its mean position, and the whole body is shifted by the planted feet's mean offset from their
pins (smoothed, so the correction never jumps). Body shape is unchanged; only the ground position moves.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

FEET = (10, 11)  # SMPL-24 left/right foot (toes)
CONTACT_Z = 0.08  # m above the lowest foot of the clip
CONTACT_V = 1.0  # m/s, smoothed foot speed
MIN_LEN = 4  # frames
FPS = 30.0


def _segments(mask):
    out, start = [], None
    for i, m in enumerate(np.r_[mask, False]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


def contacts(J, fps=FPS):
    """J (T,24,3) -> {foot joint: [(t0, t1), ...]} planted stretches."""
    out = {}
    ok = np.isfinite(J[:, 0, 0])
    floor = np.nanpercentile(J[ok][:, list(FEET), 2], 5) if ok.any() else 0.0
    for k in FEET:
        P = J[:, k]
        good = np.isfinite(P).all(1)
        Ps = P.copy()
        if good.sum() > 3:
            for c in range(3):
                Ps[:, c] = gaussian_filter1d(np.interp(np.arange(len(P)), np.flatnonzero(good), P[good, c]), 1.0)
        v = np.r_[0.0, np.linalg.norm(np.diff(Ps[:, :2], axis=0), axis=1)] * fps
        mask = good & (P[:, 2] < floor + CONTACT_Z) & (v < CONTACT_V)
        out[k] = [(a, b) for a, b in _segments(mask) if b - a >= MIN_LEN]
    return out


def lock(J, fps=FPS, sigma=1.5):
    """-> J shifted in x/y so planted feet stay put; frames without a contact keep the smoothed shift."""
    T = len(J)
    off, cnt = np.zeros((T, 2)), np.zeros(T)
    for k, segs in contacts(J, fps).items():
        for a, b in segs:
            pin = J[a:b, k, :2].mean(0)
            off[a:b] += pin - J[a:b, k, :2]
            cnt[a:b] += 1
    have = cnt > 0
    if not have.any():
        return J.copy()
    off[have] /= cnt[have, None]
    full = np.stack([np.interp(np.arange(T), np.flatnonzero(have), off[have, c]) for c in range(2)], 1)
    full = gaussian_filter1d(full, sigma, axis=0)
    out = J.copy()
    out[..., :2] += full[:, None]
    return out


ROOT_SIGMA = 4.0  # frames; DroneData test: pelvis jitter 27.6 -> 2.9 m/s^2, planted-foot slide 0.71 -> 0.21 m/s,
                  # player error vs the top-view clicks 0.215 -> 0.212 m median (p90 0.515 -> 0.483)


def smooth_root(J, sigma=ROOT_SIGMA):
    """Shift each frame's body so its pelvis follows a Gaussian-smoothed path (shape unchanged)."""
    ok = np.isfinite(J[:, 0, 0])
    if ok.sum() < 10 or sigma <= 0:
        return J.copy()
    P = np.stack([gaussian_filter1d(np.interp(np.arange(len(J)), np.flatnonzero(ok), J[ok, 0, c]), sigma)
                  for c in range(3)], 1)
    return J + (P - J[:, 0])[:, None]


def stabilize(J, fps=FPS):
    """Monocular bodies made physically plausible: smooth ground path, then planted feet pinned."""
    if np.isfinite(J[:, 0, 0]).sum() <= 10:
        return J.copy()
    return lock(smooth_root(J), fps)
