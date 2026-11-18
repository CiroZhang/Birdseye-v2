"""Per-shot features from a broadcast reconstruction, in the hitter's own frame, for the FineBadminton
stroke-type / court-zone / next-shot models (tools/fb_build_train.py, tools/fb_train.py).

Hitter frame: u = metres to the hitter's RIGHT across the court (0..6.1), d = metres from the
hitter's own baseline toward the net (0..6.7), forward velocity = toward the opponent. The bottom
(near) player faces away from the camera, the top (far) player faces it."""
import numpy as np

from .court import L, NET_Y, W

STROKE_NAMES = {"kill": "Smash", "net kill": "Net kill", "net lift": "Net lift", "push shot": "Push shot",
                "net shot": "Net shot", "cross-court net shot": "Cross-court net shot", "block": "Block",
                "drive": "Drive", "clear": "Clear", "drop shot": "Drop shot", "serve": "Serve"}
ZONES = ["left front court", "mid front court", "right front court", "left mid court", "mid court",
         "right mid court", "left back court", "mid back court", "right back court"]

SHOT_FEATURES = ["side", "u", "d", "z", "v_u", "v_f", "v_z", "speed", "elev", "apex", "dur", "end_u", "end_d",
                 "end_z", "hand_z", "anchored", "k", "prev_speed", "prev_apex", "prev_end_d", "prev_elev",
                 "next_known", "frac_obs"]


def to_hitter(xyz, side):
    x, y, z = xyz
    return (x, y, z) if side == "bottom" else (W - x, L - y, z)


def vel_to_hitter(v, side):
    return (v[0], v[1], v[2]) if side == "bottom" else (-v[0], -v[1], v[2])


def shot_rows(rec, fps, sides=None):
    """One feature dict per shot of rec['fit']. sides: optional list of 'top'/'bottom' per shot
    (labels); default = side of the hit point."""
    fit = rec["fit"]
    obs = np.asarray(rec["frames"], float)
    rows, prev = [], None
    for i, (a, b) in enumerate(fit.shots):
        ts = np.linspace(a, b, max(int((b - a) * 2), 4))
        X = fit.position(ts)
        p0 = X[0]
        side = sides[i] if sides else ("bottom" if p0[1] < NET_Y else "top")
        v = fit.params[6 * i + 3:6 * i + 6]
        u, d, z = to_hitter(p0, side)
        vu, vf, vz = vel_to_hitter(v, side)
        opp = "top" if side == "bottom" else "bottom"
        eu, ed, ez = to_hitter(X[-1], opp)
        sp = float(np.linalg.norm(v))
        hand = fit.hands[i]
        r = {"side": int(side == "top"), "u": u, "d": d, "z": z, "v_u": vu, "v_f": vf, "v_z": vz, "speed": sp,
             "elev": float(np.degrees(np.arctan2(vz, np.hypot(vu, vf)))), "apex": float(X[:, 2].max()),
             "dur": (b - a) / fps, "end_u": eu, "end_d": ed, "end_z": ez,
             "hand_z": float(hand[2]) if hand is not None else -1.0, "anchored": int(hand is not None), "k": i,
             "prev_speed": prev["speed"] if prev else -1, "prev_apex": prev["apex"] if prev else -1,
             "prev_end_d": prev["end_d"] if prev else -1, "prev_elev": prev["elev"] if prev else -99,
             "next_known": int(i < len(fit.shots) - 1),
             "frac_obs": float(((obs >= a) & (obs <= b)).sum() / max(b - a, 1)),
             "t0": float(a), "t1": float(b), "hitter": side}
        rows.append(r)
        prev = r
    return rows
