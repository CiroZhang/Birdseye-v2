"""FreeTrack's 3D step (Machine Vision and Applications 2026, github.com/Alexanderyk6/FreeTrack, road/position.py and
hit/move.py), ported to our inputs so that it sees exactly what MonoTrack sees: our calibration, our TrackNet track, our
YOLOv8-pose people and the same detected hits.

Per hit: the hitter is the person whose right wrist (COCO 10) is nearest the shuttle in the image (if the two nearest
are within 5 px, the nearest head, COCO 0, decides). The "ground projection point" is the pixel (shuttle x, mid-ankle
y of the hitter), cast to the floor z=0. Between two consecutive hits the flight lies in the vertical plane through
their two ground points; every shuttle detection in between is placed where its camera ray meets that plane. Frames
without a detection are linearly interpolated between detections; leading/trailing gaps inside a segment take the
nearest placed point (as in position.py).
"""
import numpy as np

R_WRIST, NOSE, L_ANK, R_ANK = 10, 0, 15, 16


def _ball_px(frames, px, f):
    """Shuttle pixel at frame f, or at the nearest frame with a detection (hit/move.find_nearest_valid_ball)."""
    i = int(np.argmin(np.abs(frames - f)))
    return px[i]


def _players(cam, people, margin=1.0):
    """FreeTrack keeps the on-court players (MMPose + "context-aware logic to filter non-players") and needs only that
    the head, right wrist and both ankles exist (it does not threshold their confidence). Here: people whose mid-ankle
    casts onto the court (+margin), keypoints taken as given."""
    out = []
    for kp in people:
        A = (kp[L_ANK, :2] + kp[R_ANK, :2]) / 2
        g = cam.pixel_to_ground(A[None], z=0.0)[0]
        if np.isfinite(g).all() and -margin < g[0] < 6.1 + margin and -margin < g[1] < 13.4 + margin:
            out.append(kp)
    return out


def _ground_point(cam, poses, frames, px, f):
    f = int(round(f))
    if not (0 <= f < len(poses)):
        return None
    n = _ball_px(frames, px, f)
    cand = _players(cam, poses[f])
    if not cand:  # FreeTrack falls back to the nearest frame with a complete skeleton
        for df in range(1, 13):
            for g in (f - df, f + df):
                if 0 <= g < len(poses):
                    cand = _players(cam, poses[g])
                    if cand:
                        break
            if cand:
                break
    if not cand:
        return None
    dw = np.array([np.linalg.norm(kp[R_WRIST, :2] - n) for kp in cand])
    o = np.argsort(dw)
    hit = cand[o[0]]
    if len(o) > 1 and abs(dw[o[0]] - dw[o[1]]) < 5:
        dh = [np.linalg.norm(kp[NOSE, :2] - n) for kp in cand]
        hit = cand[int(np.argmin(dh))]
    A = (hit[L_ANK, :2] + hit[R_ANK, :2]) / 2
    B = np.array([[n[0], A[1]]])
    G = cam.pixel_to_ground(B, z=0.0)[0]
    return G if np.isfinite(G).all() else None


def reconstruct(cam, frames, px, hits, poses):
    """frames (N,), px (N,2) detections; hits: increasing frame indices (segment boundaries, last = end of track).
    Returns f(t) -> (len(t), 3), NaN outside the reconstructed frames."""
    frames = np.asarray(frames, float)
    px = np.asarray(px, float)
    det = {int(round(f)): p for f, p in zip(frames, px)}
    pts = {}
    hits = [int(round(h)) for h in hits]
    G = [_ground_point(cam, poses, frames, px, h) for h in hits]
    for k in range(len(hits) - 1):
        s, e = hits[k], hits[k + 1]
        if e <= s or G[k] is None or G[k + 1] is None or np.linalg.norm(G[k + 1][:2] - G[k][:2]) < 1e-3:
            continue
        nrm = np.cross(G[k + 1] - G[k], [0.0, 0.0, 1.0])
        seg = {}
        fs = [f for f in range(s, e + 1) if f in det]
        if fs:
            c, d = cam.rays(np.array([det[f] for f in fs]))
            den = d @ nrm
            t = ((G[k] - c) @ nrm) / np.where(np.abs(den) < 1e-6, np.nan, den)
            for f, tt, dd in zip(fs, t, d):
                if np.isfinite(tt) and tt > 0:
                    seg[f] = c + tt * dd
        if not seg:
            continue
        kf = np.array(sorted(seg))
        P = np.array([seg[f] for f in kf])
        allf = np.arange(s, e + 1)
        Q = np.stack([np.interp(allf, kf, P[:, j]) for j in range(3)], 1)  # inner gaps linear, edges held
        for f, q in zip(allf, Q):
            pts[int(f)] = q
    if not pts:
        return lambda t: np.full((len(np.atleast_1d(t)), 3), np.nan)
    kf = np.array(sorted(pts))
    P = np.array([pts[f] for f in kf])

    def at(t):
        t = np.atleast_1d(np.asarray(t, float))
        out = np.full((len(t), 3), np.nan)
        ok = (t >= kf[0]) & (t <= kf[-1])
        # only inside covered frames: a missing frame (segment skipped) stays NaN
        i = np.clip(np.searchsorted(kf, t[ok]), 1, len(kf) - 1)
        a, b = kf[i - 1], kf[i]
        good = (b - a) <= 1
        w = np.where(b > a, (t[ok] - a) / np.maximum(b - a, 1e-9), 0.0)
        v = P[i - 1] + w[:, None] * (P[i] - P[i - 1])
        v[~good & (t[ok] != a) & (t[ok] != b)] = np.nan
        out[ok] = v
        return out
    return at
