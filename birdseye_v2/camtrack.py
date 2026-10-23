"""Per-frame cameras for broadcast clips that cut, zoom or pan.

A clip's single calibration (tools/fb_birdscourt.py) is only right while the camera holds still: FBBench /
FineBadminton clips cut to close-ups and replays, cut between different court cameras (back, side, corner),
and some pan or zoom. Here:

  1. shots: hard cuts from the frame-to-frame change of a small grey thumbnail
  2. every frame is checked against a camera by line precision -- the share of the visible court model
     (floor lines + net tape) that lies on painted line pixels (courtfit.line_mask)
  3. per shot: keep the clip camera where it fits; otherwise calibrate the shot (courtfit.calibrate on a
     keyframe); where the camera moves inside a shot, track it frame to frame (courtfit.refine from the
     previous frame's camera); frames no camera fits (close-ups, crowd, scoreboard) get none

track(video) -> {"cams": [cam dict], "frame_cam": [index into cams or -1 per frame], "shots": [[a, b)], ...}
"""
import cv2
import numpy as np

from . import courtfit as cf
from .court import W, L, Camera

CUT_DIFF = 30.0  # mean |grey change| (0-255) on a 64x36 thumbnail; within-shot motion is ~2-10
MIN_SHOT = 5  # frames; shorter runs (flashes, transitions) belong to the neighbouring shot
OK_PREC = 0.80  # a frame "fits" a camera when this share of the visible model is on paint (calibrate needs 0.85)
STEP = 3  # check every STEP-th frame, fill between
# low courtside shots (camera under LOW_Z, or calibrated only with the wide line mask): the near player hides much
# of the court and the camera pans fast, so frames are accepted at a lower precision and refined from a wider basin
OK_PREC_LOW, TRUNCS_LOW, LOW_Z = 0.70, (20.0, 10.0, 5.0), 2.5
HOLE_MIN = 15  # checked frames (x STEP): a longer stretch the tracked camera lost inside a shot is calibrated afresh
N_CALIB = 6  # frames of a shot tried for calibration (low courtside cameras pan, and the near player hides lines)


def read_frames(video, width=1280):
    cap, out = cv2.VideoCapture(video), []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if im.shape[1] != width:
            im = cv2.resize(im, (width, int(round(im.shape[0] * width / im.shape[1]))))
        out.append(im)
    return out


def shots(frames):
    """[(a, b)] half-open frame ranges between hard cuts."""
    th = [cv2.cvtColor(cv2.resize(f, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
          for f in frames]
    d = np.array([0.0] + [np.abs(th[i] - th[i - 1]).mean() for i in range(1, len(th))])
    cuts = [0] + [i for i in range(1, len(d)) if d[i] > CUT_DIFF] + [len(frames)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if out and b - a < MIN_SHOT:
            out[-1] = (out[-1][0], b)
        elif b > a:
            out.append((a, b))
    if len(out) > 1 and out[0][1] - out[0][0] < MIN_SHOT:
        out[1] = (0, out[1][1]); out.pop(0)
    return out


def masks(img, wide=False):
    mask = cf.line_mask(img, wide)
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
    return mask, dt


_MP = None


def precision(cam, dt):
    """Share of the visible court model on paint; 0 when under a quarter of it is in the picture."""
    global _MP
    if _MP is None:
        _MP = cf.model_points()
    h, w = dt.shape
    uv = cam.project(_MP)
    vis = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    if vis.mean() < 0.25:
        return 0.0
    return float((dt[uv[vis, 1].astype(int), uv[vis, 0].astype(int)] < cf.TOL_PX).mean())


def view_of(cam):
    c = cam.center
    return "side" if abs(c[0] - W / 2) > abs(c[1] - L / 2) else "end"


def _fit(cam, dt, truncs=(8.0, 4.0)):
    """Refine cam on this frame; (camera, precision) -- the refined one only if it fits better."""
    p0 = precision(cam, dt)
    if p0 >= 0.97:
        return cam, p0
    r = cf.refine(cam, dt, truncs=truncs)
    if not cf.camera_ok(r):
        return cam, p0
    p1 = precision(r, dt)
    return (r, p1) if p1 > p0 else (cam, p0)


def _flip(cam):
    """The same picture explained from the court's other end ((x, y) -> (W - x, L - y))."""
    A, b = np.diag([-1.0, -1.0, 1.0]), np.array([W, L, 0.0])
    return Camera(cam.f, cv2.Rodrigues(cam.R @ A)[0].ravel(), cam.R @ b + cam.tvec, cam.size, cam.dist)



def lerp(c0, c1, a):
    """Camera between two nearby ones (small motion: linear in f, rotation vector and translation)."""
    return Camera((1 - a) * c0.f + a * c1.f, (1 - a) * np.asarray(c0.rvec) + a * np.asarray(c1.rvec),
                  (1 - a) * np.asarray(c0.tvec) + a * np.asarray(c1.tvec), c0.size, c0.dist)


def track(frames, init=None, step=STEP, log=None):
    """Per-frame cameras. init: a clip-level Camera to try first (any resolution). Returns
    (cams: list of Camera or None per frame, shot list, per-shot info)."""
    n = len(frames)
    h, w = frames[0].shape[:2]
    if init is not None and init.size[0] != w:
        s = w / init.size[0]
        init = Camera(init.f * s, init.rvec, init.tvec, (w, h), init.dist)
    cams = [None] * n
    info = []
    known = [init] if init is not None else []
    for a, b in shots(frames):
        idx = list(range(a, b, step))
        if idx[-1] != b - 1:
            idx.append(b - 1)
        dts, wide = {}, [False]
        ok_prec, truncs = OK_PREC, (8.0, 4.0)

        def dt_of(i):
            if i not in dts:
                dts[i] = masks(frames[i], wide[0])[1]
            return dts[i]

        # a starting camera for this shot: a known one that fits (refined), else calibrate a frame of the shot
        start, cam0, how = None, None, None
        probe = [idx[len(idx) // 2], idx[0], idx[-1]]
        for i in probe:
            for k in known:
                c, p = _fit(k, dt_of(i))
                if p >= OK_PREC:
                    start, cam0, how = i, c, "known"
                    break
            if start is not None:
                break
        if start is None and b - a >= 8:
            tries = probe[:2] + [idx[k] for k in np.linspace(0, len(idx) - 1, N_CALIB).astype(int)]
            for i in list(dict.fromkeys(tries))[:N_CALIB]:
                c, st = cf.calibrate(frames[i])
                if c is not None:
                    if st.get("mask") == "wide":  # this shot's lines need the wide mask for every check
                        wide[0] = True
                        dts.clear()
                    if wide[0] or c.center[2] < LOW_Z:
                        ok_prec, truncs = OK_PREC_LOW, TRUNCS_LOW
                    c, p = _fit(c, dt_of(i), truncs)
                    if p >= ok_prec:
                        start, cam0, how = i, c, "calibrated"
                        break
        if start is None:
            info.append({"shot": [a, b], "cam": None})
            continue
        # track outward from the start frame, re-fitting from the neighbour's camera
        got = {start: cam0}
        j = idx.index(start) if start in idx else 0
        for order in (idx[j + 1:], idx[:j][::-1]):
            prev = cam0
            for i in order:
                c, p = _fit(prev, dt_of(i), truncs)
                if p >= ok_prec:
                    got[i] = prev = c
        # holes: a stretch of the shot the tracked camera lost -- a fade or a cut the frame difference missed, a
        # replay from another camera: calibrate inside it and track that stretch from there
        miss = [k for k, i in enumerate(idx) if i not in got]
        runs = [r for r in np.split(np.array(miss, int), np.flatnonzero(np.diff(miss) > 1) + 1) if len(r) >= HOLE_MIN] if miss else []
        for r in runs:
            for k in (r[len(r) // 2], r[len(r) // 4], r[3 * len(r) // 4]):
                c, st = cf.calibrate(frames[idx[k]])
                if c is None:
                    continue
                c, p = _fit(c, dt_of(idx[k]), truncs)
                if p < ok_prec:
                    continue
                got[idx[k]] = c
                for order in ([idx[m] for m in range(k + 1, r[-1] + 1)], [idx[m] for m in range(k - 1, r[0] - 1, -1)]):
                    prev = c
                    for i in order:
                        c2, p2 = _fit(prev, dt_of(i), truncs)
                        if p2 >= ok_prec:
                            got[i] = prev = c2
                break
        ks = sorted(got)
        for i in range(a, b):
            lo = max([k for k in ks if k <= i], default=None)
            hi = min([k for k in ks if k >= i], default=None)
            if lo is None or hi is None:
                if lo is not None and i - lo <= step:
                    cams[i] = got[lo]
                elif hi is not None and hi - i <= step:
                    cams[i] = got[hi]
                continue
            if hi - lo > 2 * step:  # lost in between (zoom-in, occlusion): leave a hole
                continue
            cams[i] = got[lo] if hi == lo else lerp(got[lo], got[hi], (i - lo) / (hi - lo))
        centres = np.array([got[k].center for k in ks])
        fs = np.array([got[k].f for k in ks])
        info.append({"shot": [a, b], "cam": how, "view": view_of(cam0), "fit": len(ks) / len(idx), "wide": wide[0],
                     "moving": bool(np.ptp(fs) / fs.mean() > 0.03 or np.ptp(centres, 0).max() > 0.5)})
        known = [cam0] + [k for k in known if k is not cam0]
        if log:
            log(info[-1])
    return cams, info


def _torso_colours(frame, cam, kps):
    """[(half, Lab colour)] of the people standing on the court: half 0 = y < NET_Y, 1 = beyond."""
    from .court import NET_Y
    from .players import players_3d
    out = []
    lab = None
    for p in players_3d(cam, kps):
        kp = p["kp"]
        t = kp[[5, 6, 11, 12]]
        if (t[:, 2] < 0.3).any():
            continue
        x0, y0 = t[:, :2].min(0).astype(int)
        x1, y1 = t[:, :2].max(0).astype(int)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        if lab is None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        out.append((int(p["foot"][1] > NET_Y), lab[y0:y1, x0:x1].reshape(-1, 3).mean(0)))
    return out


def resolve_flips(frames, cams, info, poses, every=5):
    """A shot calibrated on its own is folded to its own canonical end; the court looks the same from the other
    end, so it may be the 180-degree turned world of the clip's main shot (players swapped between halves).
    Keep each shot's frame where the players' shirt colours per court half agree with the longest shot's."""
    good = [s for s in info if s["cam"] is not None]
    if len(good) < 2 or poses is None:
        return 0
    ref = max(good, key=lambda s: s["shot"][1] - s["shot"][0])

    def half_colours(s, flip):
        acc = {0: [], 1: []}
        for i in range(s["shot"][0], s["shot"][1], every):
            if cams[i] is None or i >= len(poses):
                continue
            for half, c in _torso_colours(frames[i], _flip(cams[i]) if flip else cams[i], poses[i]):
                acc[half].append(c)
        return {k: np.median(v, 0) if v else None for k, v in acc.items()}

    rc = half_colours(ref, False)
    n = 0
    for s in good:
        if s is ref or s["cam"] != "calibrated":
            continue
        sc = half_colours(s, False)
        d = {}
        for flip in (False, True):
            pairs = [(rc[h], sc[h ^ flip]) for h in (0, 1) if rc[h] is not None and sc[h ^ flip] is not None]
            d[flip] = np.mean([np.linalg.norm(a - b) for a, b in pairs]) if pairs else np.inf
        s["flip_cost"] = [round(float(d[False]), 1), round(float(d[True]), 1)]
        if d[True] < d[False]:
            for i in range(*s["shot"]):
                if cams[i] is not None:
                    cams[i] = _flip(cams[i])
            s["flipped"] = True
            n += 1
    return n


def save(path, cams, info):
    import json
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    size = next((c.size for c in cams if c is not None), None)
    rows = [None if c is None else [round(float(c.f), 2), *np.round(np.asarray(c.rvec, float).ravel(), 6).tolist(),
                                    *np.round(np.asarray(c.tvec, float).ravel(), 5).tolist(),
                                    *np.round(np.asarray(c.dist, float).ravel(), 6).tolist()] for c in cams]
    json.dump({"size": None if size is None else list(map(int, size)), "shots": info, "frames": rows}, open(path, "w"))


def load(path, size=None):
    """-> list (per source frame) of Camera or None; size rescales (cameras are stored at 1280 wide)."""
    import json
    d = json.load(open(path))
    if d["size"] is None:
        return [None] * len(d["frames"])
    s = 1.0 if size is None else size[0] / d["size"][0]
    sz = tuple(d["size"]) if size is None else tuple(size)
    return [None if r is None else Camera(r[0] * s, r[1:4], r[4:7], sz, r[7:9]) for r in d["frames"]]


SAME_CENTRE_M = 1.5  # m: cameras this close are one physical camera (pan / tilt / zoom) for transfer()


def reference(cams):
    """The camera of the frames the longest stretch of footage shares a centre with (the main court camera),
    or None. Ties to the median-focal frame of that group."""
    idx = [i for i, c in enumerate(cams) if c is not None]
    if not idx:
        return None
    C = np.array([cams[i].center for i in idx])
    votes = (np.linalg.norm(C[:, None] - C[None], axis=2) < SAME_CENTRE_M).sum(1)
    grp = [idx[k] for k in np.flatnonzero(np.linalg.norm(C - C[int(np.argmax(votes))], axis=1) < SAME_CENTRE_M)]
    fs = np.array([cams[i].f for i in grp])
    return cams[grp[int(np.argsort(fs)[len(fs) // 2])]]


def transfer(cam, ref, uv):
    """Pixels seen by cam -> where ref would see the same rays. Exact when both share a centre (pan, tilt, zoom
    of one camera); None when they do not (a cut to another camera: the pixels carry no depth to move them)."""
    if cam is None or np.linalg.norm(cam.center - ref.center) > SAME_CENTRE_M:
        return None
    uv = np.asarray(uv, float).reshape(-1, 2)
    ok = np.isfinite(uv).all(1)
    out = np.full_like(uv, np.nan)
    if ok.any():
        c, d = cam.rays(uv[ok])
        out[ok] = ref.project(ref.center + 10.0 * d)
    return out
