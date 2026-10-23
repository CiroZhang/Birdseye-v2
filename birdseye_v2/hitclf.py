"""Learned hit detector for the back view: hand-made candidate generation + a gradient-boosted
classifier over physical/visual cues (trained on dev annotations, tools/train_hitclf.py).

Candidates: every local maximum of the image-space turn angle (gap tolerant, as in
shuttle3d.detect_hits but with a low threshold). Features per candidate:
  turn angle, image speed before/after and their ratio, samples and gap around it,
  shuttle image height, distance to the nearest player's reach box, that player's wrist speed,
  single-flight residual (does ONE drag flight explain both sides?)."""
import numpy as np

from .shuttle3d import RallyFit, near_player_px

FEATURES = ["angle", "speed_pre", "speed_post", "speed_ratio", "n_pre", "n_post", "gap", "img_y",
            "near_px", "wrist_speed", "flight_rms"]


def _candidates(frames, px, angle_deg=15, reach=8, nms=4):
    score = []
    for tau in np.arange(frames[0] + 1, frames[-1] - 1 + 1e-9, 0.5):
        pre = np.where((frames < tau) & (frames >= tau - reach))[0][-3:]
        post = np.where((frames > tau) & (frames <= tau + reach))[0][:3]
        if len(pre) < 2 or len(post) < 2:
            continue
        v1 = (px[pre[-1]] - px[pre[0]]) / max(frames[pre[-1]] - frames[pre[0]], 1e-6)
        v2 = (px[post[-1]] - px[post[0]]) / max(frames[post[-1]] - frames[post[0]], 1e-6)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1 or n2 < 1:
            continue
        ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1)))
        gap = frames[post[0]] - frames[pre[-1]]
        score.append((ang, tau, n1, n2, len(pre), len(post), gap))
    cand = []
    for s in sorted(score, reverse=True):
        if s[0] > angle_deg and all(abs(s[1] - c[1]) >= nms for c in cand):
            cand.append(s)
    return sorted(cand, key=lambda c: c[1])


def _wrist_speed(poses, f, uv):
    """Max wrist displacement (px/frame) over f-2..f+2 of the person nearest to uv."""
    if not (2 <= f < len(poses) - 2) or not len(poses[f]):
        return 0.0
    kp = poses[f]
    d = [np.linalg.norm(k[[9, 10], :2].mean(0) - uv) for k in kp]
    j = int(np.argmin(d))
    me = kp[j, [9, 10], :2]
    best = 0.0
    for g in (f - 2, f + 2):
        if not len(poses[g]):
            continue
        o = poses[g][np.argmin([np.linalg.norm(k[[5, 6], :2].mean(0) - kp[j, [5, 6], :2].mean(0)) for k in poses[g]])]
        best = max(best, float(np.linalg.norm(o[[9, 10], :2] - me, axis=1).max()) / 2)
    return best


def candidate_features(cam, frames, px, poses, H=2160.0):
    frames = np.asarray(frames, float)
    px = np.asarray(px, float)
    out = []
    for ang, tau, n1, n2, npre, npost, gap in _candidates(frames, px):
        uv = px[np.argmin(np.abs(frames - tau))]
        f = int(round(tau))
        near = near_player_px(poses[f], uv) if 0 <= f < len(poses) else 1e4
        m = np.abs(frames - tau) <= 10
        rms = 0.0
        if m.sum() >= 6:
            fr, u = frames[m], px[m]
            one = RallyFit(cam, fr, u, [(fr[0], fr[-1])], [None], use_continuity=False, use_net=False).fit()
            rms = float(np.sqrt(np.mean(np.sum((cam.project(one.position(fr)) - u) ** 2, 1))))
        out.append({"tau": float(tau), "angle": ang, "speed_pre": n1, "speed_post": n2,
                    "speed_ratio": max(n1, n2) / max(min(n1, n2), 1e-3), "n_pre": npre, "n_post": npost,
                    "gap": gap, "img_y": uv[1] / H, "near_px": min(near, 1e4),
                    "wrist_speed": _wrist_speed(poses, f, uv), "flight_rms": min(rms, 200.0)})
    return out


def detect_hits_learned(model, threshold, cam, frames, px, poses, nms=6):
    cands = candidate_features(cam, frames, px, poses)
    if not cands:
        return []
    X = np.array([[c[k] for k in FEATURES] for c in cands])
    p = model.predict_proba(X)[:, 1]
    hits = []
    for i in np.argsort(-p):
        if p[i] >= threshold and all(abs(cands[i]["tau"] - h) >= nms for h in hits):
            hits.append(cands[i]["tau"])
    return sorted(hits)
