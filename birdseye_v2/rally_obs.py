"""The only interface between the 2D detectors and the 3D model: one rally's detections as camera rays in the
court frame (birdseye_v2.court), so the model never sees pixels or which way the camera looks.

Inputs (all per video frame): a camera (one Camera, or a list with None where no court is in view), YOLO-pose
people (k,17,3), TrackNet shuttle detections (frames, px), YOLO racket boxes (k,5).

obs(...) -> dict of arrays, T = frames, P = player slots (tracked identities, up to MAX_P), R = racket slots:
  C        (T,3)      camera centre (NaN: no camera this frame)
  kp_dir   (T,P,17,3) unit ray direction through each keypoint (NaN: unseen / no camera)
  kp_conf  (T,P,17)   keypoint score (0 when unseen)
  foot     (T,P,2)    ground point under the ankles (ray cut at ankle height) -- a starting depth, not a label
  sh_dir   (T,3)      ray through the shuttle detection; sh_vis (T,)
  rk_dir   (T,R,3)    ray through each racket box centre; rk_size (T,R) box diagonal as an angle (rad); rk_conf (T,R)
  fps, size
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from .court import L, W
from .players import foot_point

MAX_P = 4
MAX_R = 4
MAX_GAP = 30  # frames a track may go unseen and still be continued
COURT_MARGIN = 1.5  # m: people standing further off the court are not players


def _cam(cams, f):
    if isinstance(cams, list):
        return cams[f] if f < len(cams) else None
    return cams


def _rays(cam, uv):
    c, d = cam.rays(np.asarray(uv, float).reshape(-1, 2))
    return c, d


def track(cams, poses, T, max_p=MAX_P, min_frac=0.15):
    """Identities across frames by ground position (Hungarian on foot distance) -> list of (T,17,3) keypoints."""
    tracks = []
    for f in range(T):
        cam = _cam(cams, f)
        people = []
        if cam is not None and f < len(poses):
            for kp in poses[f]:
                ft = foot_point(cam, kp)
                if ft is not None and -COURT_MARGIN < ft[0] < W + COURT_MARGIN and -COURT_MARGIN < ft[1] < L + COURT_MARGIN:
                    people.append((ft, kp))
        live = [k for k, t in enumerate(tracks) if f - t["last"] <= MAX_GAP]
        pairs = []
        if people and live:
            # cost = distance; a track unseen for a while may have moved further (looser gate, not a lower cost --
            # a lower cost makes stale duplicates win and identities alternate frame to frame)
            Cm = np.array([[np.linalg.norm(p[0] - tracks[k]["foot"]) for k in live] for p in people])
            gate = np.array([1.2 + 0.1 * (f - tracks[k]["last"]) for k in live])
            r, c = linear_sum_assignment(Cm)
            pairs = [(i, live[j]) for i, j in zip(r, c) if Cm[i, j] < gate[j]]
        used = {i for i, _ in pairs}
        for i, k in pairs:
            tracks[k]["kp"][f] = people[i][1]
            tracks[k]["foot"], tracks[k]["last"] = people[i][0], f
        for i, p in enumerate(people):
            if i not in used:
                tracks.append({"kp": {f: p[1]}, "foot": p[0], "last": f})
    # sparse while tracking (broadcasts have many short-lived people near the court); short pieces never matter
    tracks = [t for t in tracks if len(t["kp"]) >= 5]
    for t in tracks:
        kp = np.full((T, 17, 3), np.nan, np.float32)
        for f, k in t["kp"].items():
            kp[f] = k
        t["kp"] = kp
    tracks = _join(tracks, cams)
    seen = [np.isfinite(t["kp"][:, 0, 0]).sum() for t in tracks]
    tracks, seen = _dedupe(tracks, seen, cams)
    keep = [k for k in np.argsort(seen)[::-1][:max_p] if seen[k] >= min_frac * T]
    return [tracks[k]["kp"] for k in sorted(keep)]


def _join(tracks, cams, max_gap=150):
    """Join track pieces that do not overlap in time when one ends near where the other starts (a player lost
    for longer than MAX_GAP, or briefly mismatched)."""
    def ends(t):
        ok = np.where(np.isfinite(t["kp"][:, 0, 0]))[0]
        a, b = ok[0], ok[-1]
        fa = foot_point(_cam(cams, a), t["kp"][a])
        fb = foot_point(_cam(cams, b), t["kp"][b])
        return a, b, fa, fb
    changed = True
    while changed:
        changed = False
        info = [ends(t) for t in tracks]
        best = None
        for i, (a1, b1, _, fb1) in enumerate(info):
            for j, (a2, b2, fa2, _) in enumerate(info):
                if i == j or a2 <= b1 or a2 - b1 > max_gap or fb1 is None or fa2 is None:
                    continue
                d = np.linalg.norm(fa2 - fb1)
                if d < 1.2 + 0.03 * (a2 - b1) and (best is None or d < best[0]):
                    best = (d, i, j)
        if best:
            _, i, j = best
            ok = np.isfinite(tracks[j]["kp"][:, 0, 0])
            tracks[i]["kp"][ok] = tracks[j]["kp"][ok]
            tracks.pop(j)
            changed = True
    return tracks


def _dedupe(tracks, seen, cams, near=0.6):
    """Drop a track that, where it overlaps a bigger one in time, stands on top of it (a second skeleton on the
    same person)."""
    def feet(t):
        out = np.full((len(t["kp"]), 2), np.nan)
        for f in np.where(np.isfinite(t["kp"][:, 0, 0]))[0]:
            ft = foot_point(_cam(cams, f), t["kp"][f])
            if ft is not None:
                out[f] = ft
        return out
    F = [feet(t) for t in tracks]
    order = np.argsort(seen)[::-1]
    keep = []
    for k in order:
        dup = False
        for j in keep:
            both = np.isfinite(F[k][:, 0]) & np.isfinite(F[j][:, 0])
            if both.sum() >= 5 and np.median(np.linalg.norm(F[k][both] - F[j][both], axis=1)) < near:
                dup = True
                break
        if not dup:
            keep.append(k)
    keep = sorted(keep)
    return [tracks[k] for k in keep], [seen[k] for k in keep]


def obs(cams, poses, shuttle, rackets, T=None, fps=30.0, min_kp=0.3):
    """See the module docstring. shuttle: (frames, px) or None; rackets: list of (k,5) boxes per frame or None."""
    T = T or len(poses)
    C = np.full((T, 3), np.nan)
    for f in range(T):
        cam = _cam(cams, f)
        if cam is not None:
            C[f] = cam.center
    any_cam = next((c for c in (cams if isinstance(cams, list) else [cams]) if c is not None), None)
    P = MAX_P
    kp_dir = np.full((T, P, 17, 3), np.nan, np.float32)
    kp_conf = np.zeros((T, P, 17), np.float32)
    foot = np.full((T, P, 2), np.nan, np.float32)
    for p, kp in enumerate(track(cams, poses, T)):
        for f in np.where(np.isfinite(kp[:, 0, 0]))[0]:
            cam = _cam(cams, f)
            ok = kp[f, :, 2] > min_kp
            if not ok.any():
                continue
            _, d = _rays(cam, kp[f, ok, :2])
            kp_dir[f, p, ok] = d
            kp_conf[f, p, ok] = kp[f, ok, 2]
            ft = foot_point(cam, kp[f])
            if ft is not None:
                foot[f, p] = ft
    sh_dir = np.full((T, 3), np.nan, np.float32)
    if shuttle is not None:
        for f, uv in zip(*shuttle):
            f = int(round(f))
            cam = _cam(cams, f) if 0 <= f < T else None
            if cam is not None:
                sh_dir[f] = _rays(cam, uv)[1][0]
    rk_dir = np.full((T, MAX_R, 3), np.nan, np.float32)
    rk_size = np.zeros((T, MAX_R), np.float32)
    rk_conf = np.zeros((T, MAX_R), np.float32)
    if rackets is not None:
        for f in range(min(T, len(rackets))):
            cam = _cam(cams, f)
            b = rackets[f]
            if cam is None or not len(b):
                continue
            b = b[np.argsort(-b[:, 4])][:MAX_R]
            ctr = (b[:, :2] + b[:, 2:4]) / 2
            _, dc = _rays(cam, ctr)
            _, d0 = _rays(cam, b[:, :2])
            _, d1 = _rays(cam, b[:, 2:4])
            k = len(b)
            rk_dir[f, :k] = dc
            rk_size[f, :k] = np.arccos(np.clip((d0 * d1).sum(1), -1, 1))
            rk_conf[f, :k] = b[:, 4]
    return {"C": C.astype(np.float32), "kp_dir": kp_dir, "kp_conf": kp_conf, "foot": foot,
            "sh_dir": sh_dir, "sh_vis": np.isfinite(sh_dir[:, 0]), "rk_dir": rk_dir, "rk_size": rk_size,
            "rk_conf": rk_conf, "fps": np.float32(fps), "size": np.array(any_cam.size if any_cam else (0, 0), np.int32)}
