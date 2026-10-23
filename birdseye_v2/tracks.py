"""Temporal tracking for display: per-player identities across frames, gap filling and smoothing of
2D keypoints and foot positions, and a fixed-size racket model per player.

Offline (the whole rally is known), so gaps are filled by interpolating between the detections on
either side, and noise is removed with a centred Gaussian filter (no lag)."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import linear_sum_assignment

from .players import foot_point, lift_at_depth, players_3d

MAX_GAP = 30          # frames a joint / player may be missing and still be filled in (1 s)
SIGMA_KP = 1.5        # frames, 2D keypoint smoothing
SIGMA_FOOT = 3.0      # frames, ground position smoothing (sets the lifting depth, so it must be steady)
RACKET_LEN = 0.675    # m, typical racket (BWF max overall length 680 mm, width 230 mm)
RACKET_HEAD = (0.29, 0.22)  # m, head length x width (stringed area max 280 x 220 mm)
SIGMA_RACKET = 1.0    # frames, racket direction smoothing
RACKET_HOLD = 15      # frames a racket direction is carried across missing detections


def _fill(t, v, max_gap=MAX_GAP, hold=0):
    """Linear interpolation of v (T,D) over NaN gaps up to max_gap long; edges held for `hold` frames."""
    v = v.copy()
    ok = np.isfinite(v).all(1)
    if ok.sum() < 2:
        return v
    idx = np.where(ok)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            v[a + 1:b] = v[a] * (1 - w[:, None]) + v[b] * w[:, None]
    if hold:
        v[max(0, idx[0] - hold):idx[0]] = v[idx[0]]
        v[idx[-1] + 1:idx[-1] + 1 + hold] = v[idx[-1]]
    return v


def _smooth(v, sigma):
    """Gaussian filter over each run of finite rows separately."""
    out = v.copy()
    ok = np.isfinite(v).all(1)
    i = 0
    while i < len(v):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < len(v) and ok[j]:
            j += 1
        if j - i > 2:
            out[i:j] = gaussian_filter1d(v[i:j], sigma, axis=0, mode="nearest")
        i = j
    return out


def track_players(cam, poses, frames, max_players=4, min_frac=0.2):
    """-> list of tracks, each dict(kp=(T,17,3) raw keypoints with NaN where unseen). T = len(frames)."""
    T = len(frames)
    tracks = []  # dict(kp, last_foot, last_t)
    for i, f in enumerate(frames):
        people = players_3d(cam, poses[f]) if f < len(poses) else []
        live = [k for k, tr in enumerate(tracks) if i - tr["last_t"] <= MAX_GAP]
        if people and live:
            C = np.array([[np.linalg.norm(p["foot"] - tracks[k]["last_foot"]) - 0.3 * (i - tracks[k]["last_t"])
                           for k in live] for p in people])
            rows, cols = linear_sum_assignment(C)
            pairs = [(r, live[c]) for r, c in zip(rows, cols) if C[r, c] < 1.2]
        else:
            pairs = []
        used = {r for r, _ in pairs}
        for r, k in pairs:
            tracks[k]["kp"][i] = people[r]["kp"]
            tracks[k]["last_foot"], tracks[k]["last_t"] = people[r]["foot"], i
        for r, p in enumerate(people):
            if r not in used:
                kp = np.full((T, 17, 3), np.nan)
                kp[i] = p["kp"]
                tracks.append({"kp": kp, "last_foot": p["foot"], "last_t": i})
    seen = [np.isfinite(tr["kp"][:, 0, 0]).sum() for tr in tracks]
    keep = [k for k in np.argsort(seen)[::-1][:max_players] if seen[k] >= min_frac * T]
    return [{"kp": tracks[k]["kp"]} for k in sorted(keep)]


def smooth_track(cam, tr):
    """Adds uv (T,17,2) filled+smoothed keypoints, foot (T,2) and X (T,17,3) lifted joints."""
    kp = tr["kp"]
    T = len(kp)
    foot = np.full((T, 2), np.nan)
    for i in range(T):
        if np.isfinite(kp[i, 0, 0]):
            fp = foot_point(cam, kp[i])
            if fp is not None:
                foot[i] = fp
    foot = _smooth(_fill(np.arange(T), foot, hold=5), SIGMA_FOOT)
    uv = np.full((T, 17, 2), np.nan)
    for j in range(17):
        v = kp[:, j, :2].copy()
        v[~(kp[:, j, 2] > 0.3)] = np.nan
        uv[:, j] = _smooth(_fill(np.arange(T), v, hold=5), SIGMA_KP)
    X = np.full((T, 17, 3), np.nan)
    for i in range(T):
        if not np.isfinite(foot[i]).all():
            continue
        for j in range(17):
            if np.isfinite(uv[i, j]).all():
                p = lift_at_depth(cam, uv[i, j], foot[i])
                if p is not None:
                    X[i, j] = p
    tr.update(uv=uv, foot=foot, Xp=X.copy(), X=limb_depth(cam, X, foot))
    return tr


# segment lengths as a fraction of standing height (Winter, Biomechanics of Human Movement)
BONE_FRAC = {"upper_arm": 0.186, "forearm": 0.146, "thigh": 0.245, "shin": 0.246}
SIGMA_DEPTH = 2.0  # frames


def limb_depth(cam, X, foot):
    """Planar lifting puts every joint at the feet's depth, which flattens limbs that point toward or
    away from the camera (they look short and parallel to the body). Recover the missing depth from
    known segment lengths: a segment of true length L that appears with length l on the plane sticks
    out by sqrt(L^2 - l^2). The sign is ambiguous; arms and knees are put in front of the body
    (near-side players face away from the back camera, far-side players face it). Ankles stay put."""
    T = len(X)
    fwd = cam.R.T @ np.array([0, 0, 1.0])
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    head_z = X[:, :5, 2]
    ok = np.isfinite(head_z).any(1)
    if ok.sum() < 5:
        return X
    H = float(np.percentile(np.nanmax(head_z[ok], 1), 95)) + 0.12  # standing height (top of head)
    Lb = {k: v * H for k, v in BONE_FRAC.items()}
    # players face the net (+y on the y < 6.7 side): "in front" along fwd is the cosine between fwd and that
    # facing direction -- +-1 from an end camera, ~0 from a side camera, where the sign stays unknown and limbs
    # are left on the plane rather than pushed sideways
    face = np.where(foot[:, 1] < 6.7, 1.0, -1.0) * fwd[1]

    def ext(a, b, L):  # out-of-plane extent of segment a-b, per frame
        l = np.linalg.norm(X[:, b] - X[:, a], axis=1)
        return np.sqrt(np.clip(L ** 2 - l ** 2, 0, None))

    out = X.copy()
    for sh, el, wr in ((5, 7, 9), (6, 8, 10)):
        d_el = ext(sh, el, Lb["upper_arm"])
        d_wr = d_el + ext(el, wr, Lb["forearm"])
        for j, d in ((el, d_el), (wr, d_wr)):
            d = _smooth(d[:, None], SIGMA_DEPTH)[:, 0]
            out[:, j] += (np.nan_to_num(d) * face)[:, None] * fwd
    for hip, kn, an in ((11, 13, 15), (12, 14, 16)):
        d = 0.5 * (ext(hip, kn, Lb["thigh"]) + ext(kn, an, Lb["shin"]))
        d = _smooth(d[:, None], SIGMA_DEPTH)[:, 0]
        out[:, kn] += (np.nan_to_num(d) * face)[:, None] * fwd
    return out


def _reach_box(uv):
    k = uv[np.isfinite(uv).all(1)]
    if len(k) < 3:
        return None
    x0, y0 = k.min(0)
    x1, y1 = k.max(0)
    w, h = x1 - x0, y1 - y0
    return x0 - 0.6 * w, y0 - 0.7 * h, x1 + 0.6 * w, y1 + 0.1 * h


def rackets_for_tracks(cam, tracks, boxes_per_frame):
    """Per track: racket hand (by vote), and per frame the grip point w (T,3), unit direction u (T,3)
    and det (T,) True where a racket box was seen nearby (else the direction is carried over from
    neighbouring detections, or taken from the forearm)."""
    T = len(boxes_per_frame)
    assign = [[] for _ in tracks]  # (frame, box centre)
    for i, boxes in enumerate(boxes_per_frame):
        for bx0, by0, bx1, by1, conf in boxes:
            if conf < 0.25:
                continue
            c = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2])
            best, bd = None, 1e9
            for k, tr in enumerate(tracks):
                rb = _reach_box(tr["uv"][i])
                if rb is None or not (rb[0] <= c[0] <= rb[2] and rb[1] <= c[1] <= rb[3]):
                    continue
                d = min(np.linalg.norm(c - tr["uv"][i, j]) for j in (9, 10) if np.isfinite(tr["uv"][i, j]).all()) \
                    if np.isfinite(tr["uv"][i, [9, 10]]).any() else 1e8
                if d < bd:
                    best, bd = k, d
            if best is not None:
                assign[best].append((i, c))
    for k, tr in enumerate(tracks):
        votes = [np.argmin([np.linalg.norm(c - tr["uv"][i, j]) if np.isfinite(tr["uv"][i, j]).all() else 1e9
                            for j in (9, 10)]) for i, c in assign[k]]
        hand = (9, 10)[int(np.round(np.mean(votes)))] if votes else 10
        elbow = hand - 2
        w = tr["X"][:, hand]
        wp = tr["Xp"][:, hand]  # direction is measured on the image plane, where the box is
        u_det = np.full((T, 3), np.nan)
        for i, c in assign[k]:
            if not np.isfinite(wp[i]).all() or not np.isfinite(tr["foot"][i]).all():
                continue
            h = lift_at_depth(cam, c, tr["foot"][i])
            d = h - wp[i]
            if 0.12 < np.linalg.norm(d) < 1.0:
                u_det[i] = d / np.linalg.norm(d)
        det = np.isfinite(u_det).all(1)
        u = _fill(np.arange(T), u_det, max_gap=2 * RACKET_HOLD)
        forearm = w - tr["X"][:, elbow]
        forearm /= np.linalg.norm(forearm, axis=1, keepdims=True) + 1e-9
        # far from any detection: fall back to the forearm direction
        near = np.zeros(T, bool)
        for i in np.where(det)[0]:
            near[max(0, i - RACKET_HOLD):i + RACKET_HOLD + 1] = True
        u[~near] = forearm[~near]
        u = _smooth(u, SIGMA_RACKET)
        u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
        tr.update(racket_hand=hand, racket_w=w, racket_u=u, racket_det=det, racket_near=near)
    return tracks
