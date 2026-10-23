"""View-agnostic court calibration: back, front, side or high views, no keypoint detector.

BirdsCourt's keypoint model (and the older line search in tools/fb_calibrate.py) only knows the broadcast
view from behind a baseline; side views (camera beyond the end of the net) fail. Here:

  1. white-line mask (local top-hat, low saturation) on the court surface
  2. straight lines (merged Hough segments), grouped into the court's two line families by vanishing
     point -- not by image angle, so it does not matter which way the court runs in the picture
  3. every 2 + 2 lines of the two families x every identity they could have in the court model (either
     family may be the across-court or the along-court lines) -> a ground homography; scored on how
     much of the projected line model lies on paint, re-ranked by paint explained (precision x recall)
  4. homography -> pinhole camera (focal length from the rotation constraints), physically checked
     (above the floor, court in front, sane focal length), refined on the mask with the net tape added
  5. the court is symmetric, so the solution is canonicalised: the camera is on the y < L/2 half
     (and x < W/2 when it is beside the court) -- world frame as birdseye_v2.court

calibrate(img) -> (Camera, stats) or (None, stats).
"""
import itertools

import cv2
import numpy as np

from scipy.optimize import least_squares

from .court import COURT_SEGMENTS, L, NET_Y, POST_H, W, Camera, court_line_points

XS = [0.0, 0.46, W / 2, W - 0.46, W]  # along-court (constant x) lines; W/2 = centre line (service courts only)
YS = [0.0, 0.76, L / 2 - 1.98, L / 2 + 1.98, L - 0.76, L]  # across-court (constant y) lines
MAX_PER_FAMILY = 6
N_SEEDS = 1500  # best homographies (by precision) turned into cameras and re-ranked two-way
N_REFINE = 12  # distinct seeds refined on the mask
MIN_COURT_AREA = 0.03  # visible court footprint, share of the picture (rejects a tiny court fitted to clutter)
MIN_PRECISION = 0.85  # share of the visible model on paint; correct back-view fits: min 0.89, median 0.996
TOL_PX = 3.0  # two-way agreement tolerance at ~1280 px width
WIDE_F = 1200.0  # px at 1280 wide; below this the lens distortion is fitted too
NET_TAPE_H = 1.53  # m; 1.55 at the posts, 1.524 at the centre
# 3D segments the refinement and scoring use: the floor lines plus the net tape -- the brightest straight
# line in most frames, 1.5 m above the floor, which otherwise pulls a floor line onto itself
SEGS3D = [((x0, y0, 0.0), (x1, y1, 0.0)) for (x0, y0), (x1, y1) in COURT_SEGMENTS] +          [((0.0, NET_Y, NET_TAPE_H), (W, NET_Y, NET_TAPE_H))]


def model_points(step=0.05):
    pts = []
    for p0, p1 in SEGS3D:
        p0, p1 = np.array(p0), np.array(p1)
        n = max(2, int(np.linalg.norm(p1 - p0) / step))
        pts.append(p0 + (p1 - p0) * np.linspace(0, 1, n)[:, None])
    return np.concatenate(pts)


def refine(cam, dt, truncs=(15.0, 8.0, 4.0)):
    """Camera (f, R, t) so the projected model (floor lines + net tape) lies on paint: truncated distance-
    transform residuals, coarse to fine (as court.fit_to_mask, with the net tape)."""
    h, w = dt.shape
    pts = model_points()

    def unpack(p):
        return Camera(p[0], p[1:4], p[4:7], cam.size, p[7:9] if len(p) > 7 else cam.dist)

    def residuals(p, tr):
        uv = unpack(p).project(pts)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        r = np.full(len(pts), tr, np.float32)
        if inside.any():
            m = cv2.remap(dt, uv[inside, 0].astype(np.float32).reshape(-1, 1),
                          uv[inside, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR)
            r[inside] = np.minimum(m.ravel(), tr)
        return r

    p = np.r_[cam.f, cam.rvec, cam.tvec]
    for tr in truncs:
        p = least_squares(residuals, p, args=(tr,), x_scale="jac", diff_step=1e-4, max_nfev=300).x
    if p[0] < WIDE_F * w / 1280:  # wide lens (side / net cameras): straight lines bend, fit 2 radial terms too
        p = least_squares(residuals, np.r_[p, cam.dist], args=(truncs[-1],), x_scale="jac", diff_step=1e-4,
                          max_nfev=300).x
    return unpack(p)


def court_region(img):
    """The court surface: convex hull of the large regions of the dominant floor colour (any hue), so ad
    boards, crowd and scoreboards stay out of the line search."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    low = hsv[int(0.55 * h):]  # floor colour from under the horizon: in low courtside views crowd / boards fill the top
    sat = low[..., 1] > 40
    hue = low[..., 0][sat]
    if len(hue) < 1000:
        return np.full((h, w), 255, np.uint8)
    hist = np.bincount((hue // 6).astype(int), minlength=31)
    k = int(np.argmax(hist))
    lo, hi = max(0, 6 * k - 9), min(180, 6 * k + 15)
    g = cv2.inRange(hsv, (lo, 40, 35), (hi, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g)
    if n < 2:
        return np.full((h, w), 255, np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= 0.1 * areas.max())[0]
    pts = np.concatenate([np.stack(np.nonzero(lab == k)[::-1], 1)[::7] for k in keep]).astype(np.int32)
    out = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 255)
    return cv2.dilate(out, np.ones((25, 25), np.uint8))


def line_mask(img, wide=False):
    """Thin white structures (painted lines) on the court surface, for any floor colour: local top-hat on
    the darkest colour channel -- white paint is bright in all three, a red / green / blue floor is dark in
    at least one -- so 1-2 px far lines blended into the floor colour survive (a saturation test drops them).
    wide: also a 25 px top-hat, for low courtside cameras whose near lines are 10-15 px thick (the 11 px one
    hollows them); only a fallback -- it lets more clutter in on ordinary broadcast views."""
    mn = img.min(axis=2)
    tophat = cv2.morphologyEx(mn, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    if wide:
        tophat = np.maximum(tophat, cv2.morphologyEx(mn, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))))
    m = ((tophat > 25) & (mn > 90)).astype(np.uint8) * 255
    return cv2.bitwise_and(m, court_region(img))


def image_lines(mask, scale):
    """Merged Hough lines as homogeneous (a, b, c) with a x + b y + c = 0, |(a, b)| = 1, plus support length."""
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=int(50 * scale), minLineLength=int(50 * scale),
                           maxLineGap=int(12 * scale))
    if segs is None:
        return np.zeros((0, 3)), np.zeros(0)
    cands = []
    for x1, y1, x2, y2 in segs[:, 0]:
        d = np.array([x2 - x1, y2 - y1], float)
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        if n[1] < 0 or (n[1] == 0 and n[0] < 0):
            n = -n
        cands.append((n[0], n[1], -(n @ [x1, y1]), np.hypot(*d)))
    merged = []
    for a, b, c, ln in sorted(cands, key=lambda x: -x[3]):
        for m in merged:
            if abs(a * m[0] + b * m[1]) > np.cos(np.radians(1.5)) and abs(c * np.sign(a * m[0] + b * m[1]) - m[2]) < 5 * scale:
                m[3] += ln
                break
        else:
            merged.append([a, b, c, ln])
    merged = [m for m in merged if m[3] > 100 * scale]
    merged = sorted(merged, key=lambda m: -m[3])[:20]
    if not merged:
        return np.zeros((0, 3)), np.zeros(0)
    M = np.array(merged)
    return M[:, :3], M[:, 3]


def families(lines, support, tol_deg=2.0):
    """Split lines into two vanishing-point families: the intersection of two lines that the most line
    length passes through (within tol_deg, judged from each line's own direction) is family 1's vanishing
    point; repeat on the rest for family 2. Vanishing points at infinity (parallel lines) work too."""
    n = len(lines)
    if n < 4:
        return None
    dirs = np.c_[lines[:, 1], -lines[:, 0]]

    def consistent(vp):
        # a line belongs to vp if vp lies on it (homogeneous): angle between the line and the direction to vp
        if abs(vp[2]) < 1e-9:  # point at infinity: compare directions
            v = vp[:2] / (np.linalg.norm(vp[:2]) + 1e-12)
            cosang = np.abs(dirs @ v)
        else:
            p = vp[:2] / vp[2]
            # closest point on each line to the image centre-ish: use the foot of the perpendicular from origin
            foot = -lines[:, :2] * lines[:, 2:3]
            v = p - foot
            v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
            cosang = np.abs(np.sum(dirs * v, 1))
        return cosang > np.cos(np.radians(tol_deg))

    def best_vp(idx):
        best, bs = None, -1
        for i, j in itertools.combinations(idx, 2):
            vp = np.cross(lines[i], lines[j])
            if np.linalg.norm(vp) < 1e-12:
                continue
            vp = vp / np.linalg.norm(vp)
            m = consistent(vp)
            m[[k for k in range(n) if k not in idx]] = False
            s = support[m].sum()
            if s > bs:
                bs, best = s, m
        return best

    idx = list(range(n))
    m1 = best_vp(idx)
    if m1 is None or m1.sum() < 2:
        return None
    rest = [k for k in idx if not m1[k]]
    if len(rest) < 2:
        return None
    m2 = best_vp(rest)
    if m2 is None or m2.sum() < 2:
        return None
    f1 = [k for k in np.flatnonzero(m1)][:MAX_PER_FAMILY]
    f2 = [k for k in np.flatnonzero(m2)][:MAX_PER_FAMILY]
    return f1, f2


def _homographies(lines, f1, f2):
    """All ground->image homographies from 2 lines of each family and their possible court identities."""
    Hs = []
    for (a1, a2), (b1, b2) in itertools.product(itertools.combinations(f1, 2), itertools.combinations(f2, 2)):
        q = np.array([np.cross(lines[a1], lines[b1]), np.cross(lines[a1], lines[b2]),
                      np.cross(lines[a2], lines[b2]), np.cross(lines[a2], lines[b1])])
        if np.any(np.abs(q[:, 2]) < 1e-9):
            continue
        q = (q[:, :2] / q[:, 2:3]).astype(np.float32)
        if np.abs(q).max() > 2e4:
            continue
        for swap in (False, True):  # family 1 = across-court (y) lines, or along-court (x) lines
            for (yi, yj) in itertools.permutations(range(len(YS)), 2):
                for (xi, xj) in itertools.permutations(range(len(XS)), 2):
                    y_a, y_b, x_a, x_b = YS[yi], YS[yj], XS[xi], XS[xj]
                    if abs(y_a - y_b) < 1.5 or abs(x_a - x_b) < 1.5:
                        continue
                    if swap:  # a-lines are x = const
                        src = np.array([[x_a, y_a], [x_a, y_b], [x_b, y_b], [x_b, y_a]], np.float32)
                    else:  # a-lines are y = const
                        src = np.array([[x_a, y_a], [x_b, y_a], [x_b, y_b], [x_a, y_b]], np.float32)
                    H = cv2.getPerspectiveTransform(src, q)
                    Hs.append(H)
    return np.array(Hs) if Hs else np.zeros((0, 3, 3))


def _plausible(Hs, size):
    """Cheap physical checks on homographies: the whole court in front of the camera (no sign flip),
    a non-degenerate convex footprint of sane size."""
    w, h = size
    C = np.array([[0, 0, 1], [W, 0, 1], [W, L, 1], [0, L, 1]], float)
    P = np.einsum("nij,kj->nki", Hs, C)
    s = np.sign(P[..., 2])
    ok = np.all(s == s[:, :1], 1)
    uv = P[..., :2] / P[..., 2:3]
    area = 0.5 * np.abs(np.sum(uv[:, :, 0] * np.roll(uv[:, :, 1], -1, 1) - uv[:, :, 1] * np.roll(uv[:, :, 0], -1, 1), 1))
    ok &= (area > 0.04 * w * h) & (area < 60 * w * h)
    e = np.roll(uv, -1, 1) - uv
    cr = e[:, :, 0] * np.roll(e, -1, 1)[:, :, 1] - e[:, :, 1] * np.roll(e, -1, 1)[:, :, 0]
    ok &= np.all(cr > 0, 1) | np.all(cr < 0, 1)
    return ok


def _score(Hs, pts, dil, size, chunk=4000):
    """Precision: share of the projected line samples that land on paint, times the visible share."""
    w, h = size
    out = np.zeros(len(Hs))
    P = np.c_[pts, np.ones(len(pts))]
    for i in range(0, len(Hs), chunk):
        X = np.einsum("nij,kj->nki", Hs[i:i + chunk], P)
        z = X[..., 2]
        uv = X[..., :2] / np.where(np.abs(z) < 1e-9, 1e-9, z)[..., None]
        vis = (uv[..., 0] >= 0) & (uv[..., 0] < w - 1) & (uv[..., 1] >= 0) & (uv[..., 1] < h - 1) & (z * np.sign(z[:, :1]) > 0)
        u = np.clip(uv[..., 0], 0, w - 1).astype(int)
        v = np.clip(uv[..., 1], 0, h - 1).astype(int)
        on = dil[v, u] & vis
        frac = vis.mean(1)
        out[i:i + chunk] = np.where(frac > 0.3, on.sum(1) / np.maximum(vis.sum(1), 1) * np.sqrt(frac), 0)
    return out


def _recall(H, mask_pts, shape):
    canvas = np.zeros(shape, np.uint8)
    for (x0, y0), (x1, y1) in COURT_SEGMENTS:
        seg = cv2.perspectiveTransform(np.array([[[x0, y0]], [[x1, y1]]], np.float32), H).reshape(-1, 2)
        if not np.isfinite(seg).all() or np.abs(seg).max() > 1e5:
            continue
        cv2.line(canvas, tuple(np.round(seg[0]).astype(int)), tuple(np.round(seg[1]).astype(int)), 255, 7)
    return (canvas[mask_pts[:, 1], mask_pts[:, 0]] > 0).mean()


def two_way(cam, mask, dt, mpts):
    """(precision, recall) at TOL_PX: share of visible model line samples on paint, share of paint pixels
    (sample mpts) on a model line."""
    h, w = mask.shape
    uv = cam.project(model_points())
    vis = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    if vis.mean() < 0.25:  # under a quarter of the court lines in the picture: not enough to trust
        return 0.0, 0.0
    prec = float((dt[uv[vis, 1].astype(int), uv[vis, 0].astype(int)] < TOL_PX).mean())
    canvas = np.zeros((h, w), np.uint8)
    for p0, p1 in SEGS3D:
        q = cam.project(np.array(p0) + (np.array(p1) - np.array(p0)) * np.linspace(0, 1, 60)[:, None])
        if np.isfinite(q).all() and np.abs(q).max() < 1e5:
            cv2.polylines(canvas, [np.round(q).astype(np.int32)], False, 255, int(2 * TOL_PX + 1))
    rec = float((canvas[mpts[:, 1], mpts[:, 0]] > 0).mean())
    return prec, rec


def camera_from_homography(H, size):
    """Ground (x, y, 0) -> image homography to a pinhole camera (square pixels, centred principal point)."""
    w, h = size
    T = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1.0]])
    G = T @ H
    g1, g2 = G[:, 0], G[:, 1]
    est = []
    d = g1[2] * g2[2]
    if abs(d) > 1e-12:
        est.append(-(g1[0] * g2[0] + g1[1] * g2[1]) / d)
    d = g1[2] ** 2 - g2[2] ** 2
    if abs(d) > 1e-12:
        est.append((g2[0] ** 2 + g2[1] ** 2 - g1[0] ** 2 - g1[1] ** 2) / d)
    est = [e for e in est if e > 0]
    if not est:
        return None
    f = float(np.sqrt(np.median(est)))
    Ki = np.diag([1 / f, 1 / f, 1.0])
    r1, r2, t = Ki @ G[:, 0], Ki @ G[:, 1], Ki @ G[:, 2]
    lam = 1 / np.sqrt(np.linalg.norm(r1) * np.linalg.norm(r2))
    r1, r2, t = r1 * lam, r2 * lam, t * lam
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.stack([r1, r2, np.cross(r1, r2)], 1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    return Camera(f, cv2.Rodrigues(R)[0].ravel(), t, size)


def camera_ok(cam):
    """Physically possible broadcast/drone camera: above the floor, within a stadium, looking at the court."""
    c = cam.center
    if not (0.5 < c[2] < 80 and np.hypot(c[0] - W / 2, c[1] - L / 2) < 120 and 150 < cam.f < 12000):
        return False
    fwd = cam.R[2]  # optical axis in world
    to_court = np.array([W / 2, L / 2, 0.0]) - c
    return fwd @ to_court / np.linalg.norm(to_court) > 0.3


def canonical(cam):
    """The court looks the same rotated 180 deg about its centre ((x, y) -> (W - x, L - y); a mirror would
    flip handedness, so it is the only symmetry a camera can take). Fold it: an end-on camera sits on the
    y < L/2 half (as the back-view convention), a side camera on the x < W/2 side."""
    c = cam.center
    side = abs(c[0] - W / 2) > abs(c[1] - L / 2)
    if not (c[0] > W / 2 if side else c[1] > L / 2):
        return cam
    A, b = np.diag([-1.0, -1.0, 1.0]), np.array([W, L, 0.0])  # X_old = A X_new + b
    R = cam.R
    return Camera(cam.f, cv2.Rodrigues(R @ A)[0].ravel(), R @ b + cam.tvec, cam.size, cam.dist)


def calibrate(img, do_refine=True):
    """img (BGR) -> (Camera or None, stats). Works at any resolution (internally ~1280 wide). Retries with the
    wide line mask when the ordinary one finds no camera."""
    cam, stats = _calibrate(img, do_refine, wide=False)
    if cam is None:
        cam2, stats2 = _calibrate(img, do_refine, wide=True)
        if cam2 is not None:
            return cam2, {**stats2, "mask": "wide"}
    return cam, stats


def _calibrate(img, do_refine, wide):
    h0, w0 = img.shape[:2]
    s = 1280.0 / w0
    im = cv2.resize(img, (1280, int(round(h0 * s)))) if abs(s - 1) > 1e-3 else img
    size = (im.shape[1], im.shape[0])
    mask = line_mask(im, wide)
    lines, sup = image_lines(mask, 1.0)
    stats = {"n_lines": int(len(lines))}
    fam = families(lines, sup) if len(lines) >= 4 else None
    if fam is None:
        return None, {**stats, "reason": "no two line families"}
    Hs = _homographies(lines, *fam)
    stats["n_hyp"] = int(len(Hs))
    if not len(Hs):
        return None, {**stats, "reason": "no hypotheses"}
    Hs = Hs[_plausible(Hs, size)]
    if not len(Hs):
        return None, {**stats, "reason": "no plausible hypotheses"}
    dil = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    pts = court_line_points(step=0.1)[:, :2]
    sc = _score(Hs, pts, dil, size)
    order = np.argsort(-sc)[:N_SEEDS]
    ys_, xs_ = np.nonzero(mask)
    sel = np.random.default_rng(0).choice(len(ys_), size=min(4000, len(ys_)), replace=False)
    mpts = np.stack([xs_[sel], ys_[sel]], 1)
    seeds = []
    for k in order:
        cam = camera_from_homography(Hs[k], size)
        if cam is None or not camera_ok(cam):
            continue
        rec = _recall(Hs[k], mpts, mask.shape)
        seeds.append((2 * sc[k] * rec / max(sc[k] + rec, 1e-9), cam))
    if not seeds:
        return None, {**stats, "reason": "no physically valid camera"}
    # refine the best few distinct seeds; keep the one whose lines lie on paint AND explain the paint
    seeds.sort(key=lambda x: -x[0])
    picked = []
    for fsc, cam in seeds:
        if all(np.linalg.norm(cam.center - c.center) > 0.5 for _, c in picked):
            picked.append((fsc, cam))
        if len(picked) >= N_REFINE:
            break
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
    best, bs = None, -1
    for fsc, cam in picked:
        if do_refine:
            cam = refine(cam, dt)
            if not camera_ok(cam):
                continue
        prec, rec = two_way(cam, mask, dt, mpts)
        f2 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f2 > bs:
            bs, best = f2, (cam, prec, rec, fsc)
    if best is None:
        return None, {**stats, "reason": "refined camera implausible"}
    cam, prec, rec, fsc = best
    foot = cam.project(np.array([[0, 0, 0], [W, 0, 0], [W, L, 0], [0, L, 0]], float)).astype(np.float32)
    clip = cv2.intersectConvexConvex(foot, np.array([[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]], np.float32))[0]
    if clip < MIN_COURT_AREA * size[0] * size[1]:
        return None, {**stats, "reason": "court too small in the picture"}
    if prec < MIN_PRECISION:
        return None, {**stats, "reason": f"model lines not on paint (precision {prec:.2f})"}
    stats.update(seed_score=round(float(fsc), 3), precision=round(float(prec), 3), recall=round(float(rec), 3),
                 score=round(float(bs), 3))
    cam = canonical(cam)
    if abs(s - 1) > 1e-3:  # back to the input resolution
        cam = Camera(cam.f / s, cam.rvec, cam.tvec, (w0, h0), cam.dist)
    c = cam.center
    stats["view"] = "side" if abs(c[0] - W / 2) > abs(c[1] - L / 2) else "end"
    stats["center"] = [round(float(v), 2) for v in c]
    stats["f"] = round(cam.f, 1)
    return cam, stats
