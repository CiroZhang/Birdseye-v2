"""Virtual cameras (training only): the multi-camera teacher's 3D rally -- TRAM bodies and shuttle -- seen from a
camera that never filmed it, as the detections that camera would have given (rally_obs arrays, same slots).

The student sees only rays from a calibrated camera, so nothing in it is tied to the TV camera except what it
learned from TV rallies. Every real training rally is broadcast footage from high behind the near baseline; the
depth along a ray from anywhere else is out of distribution (DroneData's back drone: the error lies along its
rays). Re-viewing the teacher's labels from sampled cameras -- low and close, high and far, from the corners --
teaches depth from geometry and flight, not from one viewpoint. Labels (3D, hits) are camera-free and stay as they are.

Detector imitation: pixel noise relative to the person's size (YOLO-pose) or ~2 px (TrackNet), keypoints and
shuttle dropped at random and in bursts, everything outside the image unseen. No racket boxes are made."""
import cv2
import numpy as np

from .court import L, W, Camera
from .players import foot_point
from .rally_obs import MAX_R

# COCO-17 body joints from SMPL-24 (the face is built from the SMPL head and shoulders below)
_SMPL_OF = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}
_ARM = [7, 8, 9, 10]  # elbows and wrists: the detector's least sure joints
SIZE = (1920, 1080)


def sample_camera(rng):
    """A camera behind or beside the near half of the court, looking at the court: azimuth within 75 deg of the
    TV's axis, 8-26 m from the court centre, 2.5-9 m up (TV ~ 23 m / 5 m; DroneData back drone ~ 11 m / 3.6 m),
    zoomed to hold the court with a random margin (sometimes cropping its corners)."""
    ctr = np.array([W / 2, L / 2, 0.0])
    az, dist, h = rng.uniform(-1.31, 1.31), rng.uniform(8.0, 26.0), rng.uniform(2.5, 9.0)
    C = ctr + np.array([dist * np.sin(az), -dist * np.cos(az), 0.0])
    C[2] = h
    tgt = ctr + np.array([rng.uniform(-1, 1), rng.uniform(-1.5, 1.5), rng.uniform(0.0, 1.5)])
    z = tgt - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z])
    pts = np.array([[a, b, c] for a in (0, W) for b in (0, L) for c in (0.0, 2.5)]) - C
    pc = pts @ R.T
    pc = pc[pc[:, 2] > 0.5]
    tan = np.abs(pc[:, :2] / pc[:, 2:])
    f = min(SIZE[0] / 2 / tan[:, 0].max(), SIZE[1] / 2 / tan[:, 1].max()) * rng.uniform(0.9, 1.25)
    return Camera(f, cv2.Rodrigues(R)[0], -R @ C, SIZE)


def _coco(J24):
    """(...,24,3) SMPL joints -> (...,17,3) COCO joints; nose, eyes and ears placed on the head from its facing."""
    out = np.zeros(J24.shape[:-2] + (17, 3))
    for c, s in _SMPL_OF.items():
        out[..., c, :] = J24[..., s, :]
    H = J24[..., 15, :]
    side = J24[..., 16, :] - J24[..., 17, :]  # person's right -> left
    side /= np.linalg.norm(side, axis=-1, keepdims=True) + 1e-9
    fwd = np.cross(side, [0.0, 0.0, 1.0])
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9
    up = np.array([0.0, 0.0, 1.0])
    out[..., 0, :] = H + 0.09 * fwd - 0.02 * up
    out[..., 1, :] = H + 0.07 * fwd + 0.03 * side + 0.02 * up
    out[..., 2, :] = H + 0.07 * fwd - 0.03 * side + 0.02 * up
    out[..., 3, :] = H + 0.075 * side
    out[..., 4, :] = H - 0.075 * side
    return out, fwd


def _bursts(rng, T, n, lo, hi):
    m = np.ones(T, bool)
    for _ in range(n):
        a = int(rng.integers(0, max(1, T)))
        m[a:a + int(rng.integers(lo, hi))] = False
    return m


def synth(B24, B24_mask, S, S_mask, fps, rng, min_kp=0.3):
    """Teacher bodies (T,P,24,3)+(T,P) and shuttle (T,3)+(T,) -> (rally_obs dict, camera), or None when the virtual
    camera sees too little of the rally."""
    T, P = B24_mask.shape
    cam = sample_camera(rng)
    w, h = cam.size
    Cc = cam.center
    kp_dir = np.full((T, P, 17, 3), np.nan, np.float32)
    kp_conf = np.zeros((T, P, 17), np.float32)
    foot = np.full((T, P, 2), np.nan, np.float32)
    for p in range(P):
        fr = np.where(B24_mask[:, p] > 0)[0]
        if not len(fr):
            continue
        X, fwd = _coco(B24[fr, p].astype(float))  # (n,17,3)
        uv = cam.project(X.reshape(-1, 3)).reshape(len(fr), 17, 2)
        zc = (X - Cc) @ cam.R[2]
        hpx = np.linalg.norm(uv[:, 0] - uv[:, [15, 16]].mean(1), axis=-1)  # nose to ankles, px
        sig = 0.02 * hpx[:, None] * rng.uniform(0.6, 1.4)
        sig = np.repeat(sig, 17, 1)
        sig[:, _ARM] *= 1.6
        uv = uv + rng.normal(size=uv.shape) * sig[..., None]
        face = ((Cc - X[:, 0]) * fwd).sum(-1) / (np.linalg.norm(Cc - X[:, 0], axis=-1) + 1e-9)  # >0: face towards camera
        conf = rng.uniform(0.55, 0.95, size=(len(fr), 17))
        conf[:, :5] *= np.clip(0.55 + 0.45 * face, 0.2, 1.0)[:, None]
        seen = (zc > 0.3) & (uv[..., 0] >= 0) & (uv[..., 0] < w) & (uv[..., 1] >= 0) & (uv[..., 1] < h)
        seen &= rng.random((len(fr), 17)) > 0.08  # single keypoints missed
        seen &= _bursts(rng, T, 2, 3, 15)[fr][:, None]  # the whole person missed for a few frames
        conf = np.where(seen, conf, 0.0)
        for i, t in enumerate(fr):
            ok = conf[i] > min_kp
            if not ok.any():
                continue
            kp_dir[t, p, ok] = cam.rays(uv[i, ok])[1]
            kp_conf[t, p, ok] = conf[i, ok]
            ft = foot_point(cam, np.c_[uv[i], conf[i]])
            if ft is not None:
                foot[t, p] = ft
    sh_dir = np.full((T, 3), np.nan, np.float32)
    fs = np.where((S_mask > 0) & np.isfinite(S).all(1))[0]
    if len(fs):
        uv = cam.project(S[fs].astype(float)) + rng.normal(size=(len(fs), 2)) * rng.uniform(1.0, 3.0)
        zc = (S[fs] - Cc) @ cam.R[2]
        ok = (zc > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        ok &= rng.random(len(fs)) > 0.1
        ok &= _bursts(rng, T, max(1, T // 150), 4, 20)[fs]
        if ok.any():
            sh_dir[fs[ok]] = cam.rays(uv[ok])[1]
    if (kp_conf.max(-1) > 0).sum() < 40 or np.isfinite(sh_dir[:, 0]).sum() < 20:
        return None
    o = {"C": np.repeat(Cc[None].astype(np.float32), T, 0), "kp_dir": kp_dir, "kp_conf": kp_conf, "foot": foot,
         "sh_dir": sh_dir, "sh_vis": np.isfinite(sh_dir[:, 0]), "rk_dir": np.full((T, MAX_R, 3), np.nan, np.float32),
         "rk_size": np.zeros((T, MAX_R), np.float32), "rk_conf": np.zeros((T, MAX_R), np.float32),
         "fps": np.float32(fps), "size": np.array(SIZE, np.int32)}
    return o, cam
