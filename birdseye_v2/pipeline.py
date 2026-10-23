"""Birdseye-v2 back-view shuttle reconstruction, end to end (one rally).

Inputs (back view only): calibrated camera, refined TrackNet track, YOLO pose per frame,
optional racket boxes, optional known hit times. Steps:
  1. landing detection + static-clutter removal          (shuttle3d.landing_frame, clean_track)
  2. hits: given, or detected (gap-tolerant turn + player gate + single-flight veto), then
     split shots no single flight explains / merge shots one flight explains
  3. hitter anchors: hand (or racket head) of the hitter; court side alternates per hit; the
     side assignment and within-side choice are made by fit cost
  4. joint rally fit (shuttle3d.RallyFit): robust reprojection + continuity + anchors + floor
     + landing (the last shot comes down above where the shuttle comes to rest)
"""
import numpy as np

from .players import players_3d, racket_anchor
from .shuttle3d import (FPS, clean_track, detect_hits, fit_rally, hand_candidates, landing_frame,
                        merge_explained, near_player_px, segment_shots, split_unexplained)

SEG_RMS = 16.0  # px, split/merge threshold (dev sweep: plateau from 12)


def _interp(frames, px, t):
    i = np.searchsorted(frames, t)
    if i <= 0:
        return px[0]
    if i >= len(frames):
        return px[-1]
    a = (t - frames[i - 1]) / max(frames[i] - frames[i - 1], 1e-9)
    return px[i - 1] * (1 - a) + px[i] * a


def _candidates(cam, poses, frames, px, t, rackets=None):
    f = int(round(t))
    if not (0 <= f < len(poses)):
        return []
    uv = _interp(frames, px, t)
    people = players_3d(cam, poses[f])
    cands = hand_candidates(people, uv, cam, max_px=1e9)
    if rackets is not None and f < len(rackets):
        for c in cands:
            for p in people:
                if any(np.allclose(c["hand"], w) for w in p["wrists"]):
                    ra = racket_anchor(cam, p, rackets[f], uv)
                    if ra is not None:
                        c["hand"] = ra
                    break
    return cands


def reconstruct_rally(cam, frames, px, poses, hits=None, rackets=None, hit_model=None):
    """Returns dict(fit=RallyFit, shots=[(t0,t1)], hits=[...], landing=frame|None,
    frames=..., px=... (cleaned track)). Times are back-view frame numbers."""
    frames = np.asarray(frames, float)
    px = np.asarray(px, float).reshape(-1, 2)
    detected = hits is None
    lf = landing_frame(frames, px)
    if detected:
        f_c, p_c = clean_track(frames, px, [lf - 1] if lf is not None else [frames[-1]])
        if hit_model is not None:  # learned detector (birdseye_v2.hitclf, tools/train_hitclf.py)
            from .hitclf import detect_hits_learned
            hits = detect_hits_learned(hit_model["model"], hit_model["threshold"], cam, f_c, p_c, poses)
        else:
            hits = list(detect_hits(f_c, p_c, cam=cam, poses=poses))
        if lf is not None:
            hits = [h for h in hits if h < lf - 2]
        if not hits or hits[0] > f_c[0] + 4:  # flight already under way when the track starts
            hits = [float(f_c[0])] + hits
    frames, px = clean_track(frames, px, hits)
    shots = segment_shots(frames, hits, px)
    if detected:
        shots = merge_explained(cam, frames, px, split_unexplained(cam, frames, px, shots, max_rms=SEG_RMS),
                                max_rms=SEG_RMS)
    cands = [_candidates(cam, poses, frames, px, a, rackets) for a, _ in shots]
    if detected:  # a shot start that is not next to a player is not a hit: no anchor
        for i, (a, _) in enumerate(shots):
            f = int(round(a))
            if not (0 <= f < len(poses)) or near_player_px(poses[f], _interp(frames, px, a)) > 80:
                cands[i] = []
    fit = fit_rally(cam, frames, px, shots, cands, land=True)
    return {"fit": fit, "shots": shots, "hits": [s[0] for s in shots], "landing": lf,
            "frames": frames, "px": px}


def rally_summary(rec, fps=FPS):
    """Per-shot 3D facts from a reconstruction (used by the QA layer and the demo)."""
    fit = rec["fit"]
    out = []
    for i, (a, b) in enumerate(fit.shots):
        ts = np.linspace(a, b, max(int((b - a) * 2), 4))
        X = fit.position(ts)
        v0 = fit.params[6 * i + 3:6 * i + 6]
        out.append({
            "shot": i + 1, "t_start_s": round(a / fps, 3), "t_end_s": round(b / fps, 3),
            "hit_xyz_m": np.round(X[0], 2).tolist(), "end_xyz_m": np.round(X[-1], 2).tolist(),
            "apex_m": round(float(X[:, 2].max()), 2),
            "speed_kmh": round(float(np.linalg.norm(v0)) * 3.6, 1),
            "hitter_side": "near" if X[0][1] < 6.7 else "far",
            "anchored": fit.hands[i] is not None,
        })
    return out
