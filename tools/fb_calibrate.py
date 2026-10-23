"""Calibrate the broadcast camera of FBBench clips (FineBadminton benchmark) by fitting the court line
model to a white-line mask (birdseye_v2.court.fit_to_mask). Seeded from hand-read outer corners of
one BWF broadcast frame; broadcast framing barely changes between tournaments.

Broadcast clips can contain cuts (close-ups, replays): every frame gets an 'on_court' flag = the
fraction of projected court-line samples that land on white-line pixels in that frame.

Output: outputs/fb/calib/<video_uid>.json  {cam, stats, on_court: [frac per frame]}
        outputs/fb/calib/<video_uid>.jpg   overlay for checking
usage: python tools/fb_calibrate.py video_000651 ...   (no args = all clips)"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import COURT_SEGMENTS, Camera, court_line_points, draw_court, fit_to_mask, init_from_corners

# FB_SET=bench (FBBench clips) or train (FineBadminton-20K rally clips, tools/fb20k_prep.py)
FB_SET = os.environ.get("FB_SET", "bench")
CLIPS = os.path.join(ROOT, "data", "fbbench", "hf_video_clips_qa") if FB_SET == "bench" else os.path.join(ROOT, "data", "fb20k", "clips")
OUT = os.path.join(ROOT, "outputs", "fb" if FB_SET == "bench" else "fb20k", "calib")
SIZE = (1280, 720)
# outer doubles corners in COURT_CORNERS order (near-left, near-right, far-right, far-left),
# read off video_000651 frame 0 (HSBC BWF World Tour broadcast)
SEED_CORNERS = np.array([(219, 645), (1062, 648), (866, 278), (414, 278)], float)
SEED = init_from_corners(SEED_CORNERS, SIZE, 1500.0)


def _lines(mask):
    """Long straight line candidates (merged Hough segments) as (a, b, c) with ax + by = c, |(a,b)| = 1."""
    segs = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=60, minLineLength=60, maxLineGap=12)
    if segs is None:
        return [], []
    cands = []
    for x1, y1, x2, y2 in segs[:, 0]:
        d = np.array([x2 - x1, y2 - y1], float)
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        if n[1] < 0:
            n = -n
        cands.append((n[0], n[1], n @ [x1, y1], np.hypot(*d)))
    merged = []  # greedy merge of near-identical lines, weighted by segment length
    for a, b, c, ln in sorted(cands, key=lambda x: -x[3]):
        for m in merged:
            if abs(a * m[0] + b * m[1]) > np.cos(np.radians(2)) and abs(c - m[2]) < 6:
                m[3] += ln
                break
        else:
            merged.append([a, b, c, ln])
    merged = [m for m in merged if m[3] > 120]
    horiz = [m for m in merged if abs(m[1]) > np.cos(np.radians(15))]   # normal ~ vertical
    side = [m for m in merged if abs(m[1]) <= np.cos(np.radians(25))]
    horiz = sorted(horiz, key=lambda m: -m[3])[:8]
    side = sorted(side, key=lambda m: -m[3])[:8]
    return horiz, side


def _meet(l1, l2):
    A = np.array([[l1[0], l1[1]], [l2[0], l2[1]]])
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    return np.linalg.solve(A, [l1[2], l2[2]])


def line_seed(mask):
    """MonoTrack-style: pick 2 horizontal + 2 side image lines, try every identity they could have in
    the court model, score each homography on the full line model."""
    from birdseye_v2.court import W, L
    horiz, side = _lines(mask)
    if len(horiz) < 2 or len(side) < 2:
        return None, -1
    ys = [0.0, 0.76, L / 2 - 1.98, L / 2 + 1.98, L - 0.76, L]
    xs = [0.0, 0.46, W - 0.46, W]
    pts = court_line_points(step=0.1)[:, :2].reshape(-1, 1, 2).astype(np.float32)
    dil = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    h, w = mask.shape
    cand = []
    for i in range(len(horiz)):
        for j in range(i + 1, len(horiz)):
            near, far = sorted([horiz[i], horiz[j]], key=lambda m: -m[2] / m[1])  # larger y = nearer
            for k in range(len(side)):
                for l in range(k + 1, len(side)):
                    q = [_meet(near, side[k]), _meet(near, side[l]), _meet(far, side[l]), _meet(far, side[k])]
                    if any(x is None for x in q):
                        continue
                    q = np.array(q)
                    if q[0, 0] > q[1, 0]:  # left/right by x on the near line
                        q = q[[1, 0, 3, 2]]
                    if np.abs(q).max() > 5000:
                        continue
                    for yi in range(len(ys)):
                        for yj in range(yi + 1, len(ys)):
                            for xi in range(len(xs)):
                                for xj in range(xi + 1, len(xs)):
                                    if xs[xj] - xs[xi] < 4 or ys[yj] - ys[yi] < 3:
                                        continue
                                    src = np.array([[xs[xi], ys[yi]], [xs[xj], ys[yi]], [xs[xj], ys[yj]], [xs[xi], ys[yj]]], np.float32)
                                    H = cv2.getPerspectiveTransform(src, q.astype(np.float32))
                                    rect4 = np.array([[0, 0], [W, 0], [W, L], [0, L]], np.float32).reshape(-1, 1, 2)
                                    if not court_quad_ok(cv2.perspectiveTransform(rect4, H).reshape(-1, 2)):
                                        continue
                                    uv = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
                                    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                                    if ok.mean() < 0.8:
                                        continue
                                    sc = dil[uv[ok, 1].astype(int), uv[ok, 0].astype(int)].mean() * ok.mean()
                                    cand.append((sc, H))
    if not cand:
        return None, -1
    # re-rank the best candidates by a two-way score: model lines on paint (precision) AND the court's
    # paint explained by the model (recall). Precision alone prefers a shrunken court on inner lines.
    ys_, xs_ = np.nonzero(mask)
    sel = np.random.default_rng(0).choice(len(ys_), size=min(4000, len(ys_)), replace=False)
    mpts = np.stack([xs_[sel], ys_[sel]], 1)
    best, bs = None, -1
    for sc, H in sorted(cand, key=lambda x: -x[0])[:30]:
        canvas = np.zeros(mask.shape, np.uint8)
        for (x0, y0), (x1, y1) in COURT_SEGMENTS:
            seg = cv2.perspectiveTransform(np.array([[[x0, y0]], [[x1, y1]]], np.float32), H).reshape(-1, 2)
            cv2.line(canvas, tuple(np.round(seg[0]).astype(int)), tuple(np.round(seg[1]).astype(int)), 255, 7)
        rec = (canvas[mpts[:, 1], mpts[:, 0]] > 0).mean()
        f = 2 * sc * rec / max(sc + rec, 1e-9)
        if f > bs:
            bs, best = f, H
    cam = camera_from_homography(best)
    if cam is None or not camera_ok(cam):
        rect = np.array([[0, 0], [W, 0], [W, L], [0, L]], np.float32).reshape(-1, 1, 2)
        cam = init_from_corners(cv2.perspectiveTransform(rect, best).reshape(-1, 2), SIZE, 1500.0)
    return cam, bs


def camera_from_homography(H, size=SIZE):
    """Ground plane (x, y, 0) -> image homography to a camera: focal length from the two orthogonality
    constraints on the rotation columns (square pixels, principal point at the image centre), then
    rotation / translation by decomposition (Zhang-style single-view calibration)."""
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


def court_quad_ok(C):
    """Projected outer corners (near-left, near-right, far-right, far-left) look like a broadcast court:
    a convex trapezoid, near baseline wide and below the far one. Rejects the degenerate homographies
    that squash the court onto one painted line (they score well on line overlap)."""
    C = np.asarray(C, float)
    if not np.isfinite(C).all():
        return False
    near_w, far_w = C[1, 0] - C[0, 0], C[2, 0] - C[3, 0]
    near_y, far_y = (C[0, 1] + C[1, 1]) / 2, (C[2, 1] + C[3, 1]) / 2
    area = 0.5 * abs(np.dot(C[:, 0], np.roll(C[:, 1], 1)) - np.dot(C[:, 1], np.roll(C[:, 0], 1)))
    return near_w > 300 and far_w > 120 and near_w > far_w and near_y - far_y > 120 and area > 6e4


def camera_ok(cam):
    """Broadcast camera behind and above the near baseline (world: y=0 near baseline, z up)."""
    c = cam.center
    return -6 < c[0] < 12 and -80 < c[1] < -1 and 2 < c[2] < 50 and 300 < cam.f < 6000


def search_seed(mask):
    """Broadcast framings differ in zoom and position: scan scaled / shifted / re-tilted versions of
    the reference outer corners (a ground-plane homography each) and keep the one whose projected
    court lines land on the most white-line pixels."""
    from birdseye_v2.court import W, L
    pts = court_line_points(step=0.1)[:, :2]
    rect = np.array([[0, 0], [W, 0], [W, L], [0, L]], np.float32)
    dil = cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0
    h, w = mask.shape
    c0 = SEED_CORNERS.mean(0)
    best, bs = None, -1
    for s in np.linspace(0.8, 1.5, 15):
        for t in (0.85, 0.92, 1.0, 1.08, 1.16):  # far-baseline width relative to near (camera tilt)
            for v in (0.85, 0.92, 1.0, 1.08, 1.16):  # court depth in the image
                base = SEED_CORNERS - c0
                base = base * [1, v]
                base[2:] = base[2:] * [t, 1]
                for dy in range(-120, 121, 20):
                    for dx in range(-80, 81, 20):
                        C = c0 + s * base + [dx, dy]
                        if not court_quad_ok(C):
                            continue
                        H = cv2.getPerspectiveTransform(rect, C.astype(np.float32))
                        uv = cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
                        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
                        if ok.mean() < 0.9:
                            continue
                        sc = dil[uv[ok, 1].astype(int), uv[ok, 0].astype(int)].mean()
                        if sc > bs:
                            bs, best = sc, C
    return init_from_corners(best, SIZE, 1500.0), bs


def court_region(img):
    """The court surface: convex hull of all large court-coloured (green) regions -- the net splits the
    court in two and the far half is often darker -- widened a little so the outer lines are inside.
    Keeps ad boards / banners / scoreboards out of the line search and off-court people out of play."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    g = cv2.inRange(hsv, (35, 40, 35), (95, 255, 255))
    g = cv2.morphologyEx(g, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(g)
    if n < 2:
        return np.full(g.shape, 255, np.uint8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + np.where(areas >= 0.1 * areas.max())[0]
    pts = np.concatenate([np.stack(np.nonzero(lab == k)[::-1], 1)[::7] for k in keep]).astype(np.int32)
    out = np.zeros(g.shape, np.uint8)
    cv2.fillConvexPoly(out, cv2.convexHull(pts), 255)
    return cv2.dilate(out, np.ones((15, 15), np.uint8))


def white_mask(img):
    """Thin bright low-saturation structures (painted lines) on the court surface: local contrast
    (top-hat), not absolute brightness -- far / horizontal lines are 1-2 px wide and blend in."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lowsat = cv2.inRange(hsv, (0, 0, 110), (180, 90, 255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    m = cv2.bitwise_and(lowsat, (tophat > 18).astype(np.uint8) * 255)
    return cv2.bitwise_and(m, court_region(img))


def frames(path, step=1):
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % step == 0:
            yield i, im
        i += 1


def on_court_fraction(cam, mask, pts_uv):
    h, w = mask.shape
    ok = (pts_uv[:, 0] >= 0) & (pts_uv[:, 0] < w) & (pts_uv[:, 1] >= 0) & (pts_uv[:, 1] < h)
    uv = pts_uv[ok].astype(int)
    dil = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    return float((dil[uv[:, 1], uv[:, 0]] > 0).mean()) if len(uv) else 0.0


def calibrate(uid):
    try:
        path = os.path.join(CLIPS, uid + ".mp4")
        # pass 1 (streamed, low-res): court-view frames = most green court surface in view; seed-free.
        # Clips can be a minute long: never hold all full-res frames (12 workers ran the PC out of RAM).
        green = np.array([cv2.inRange(cv2.cvtColor(cv2.resize(im, (320, 180)), cv2.COLOR_BGR2HSV),
                                      (35, 60, 60), (95, 255, 255)).mean() / 255 for _, im in frames(path)])
        if not len(green):
            return uid, None
        pts = court_line_points()
        good = np.where(green >= 0.85 * green.max())[0]
        sel = set(good[np.linspace(0, len(good) - 1, min(15, len(good))).astype(int)].tolist())
        med = np.median(np.stack([im for i, im in frames(path) if i in sel]), 0).astype(np.uint8)
        mask = white_mask(med)
        seed, seed_score = search_seed(mask)
        lseed, lscore = line_seed(mask)
        fits = []
        for s0 in [x for x in (lseed, seed, SEED) if x is not None]:  # multi-start: searched seed and the reference framing
            c1, _ = fit_to_mask(s0, mask, fit_f=True, fit_dist=False, truncs=(30.0, 15.0, 8.0))
            c1, st1 = fit_to_mask(c1, mask, fit_f=True, fit_dist=True, truncs=(8.0, 4.0))
            from birdseye_v2.court import COURT_CORNERS
            valid = camera_ok(c1) and court_quad_ok(c1.project(COURT_CORNERS))
            fits.append((valid, on_court_fraction(c1, mask, c1.project(pts)), c1, st1))
        valid, _, cam, st = max(fits, key=lambda x: (x[0], x[1]))
        st["valid"] = bool(valid)
        uv = cam.project(pts)
        on = [round(on_court_fraction(cam, white_mask(im), uv), 3) for _, im in frames(path)]  # pass 3, streamed
        st["on_court_median"] = float(np.median(on))
        st["seed_score"] = float(seed_score)
        st["line_seed_score"] = float(lscore)
        os.makedirs(OUT, exist_ok=True)
        cv2.imwrite(os.path.join(OUT, uid + ".jpg"), draw_court(med, cam, thick=1))
        json.dump({"cam": cam.to_dict(), "stats": st, "on_court": on}, open(os.path.join(OUT, uid + ".json"), "w"))
        return uid, st
    except Exception as e:
        return uid, repr(e)


if __name__ == "__main__":
    uids = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CLIPS, "*.mp4")))
    uids = [u for u in uids if not os.path.exists(os.path.join(OUT, u + ".json"))]
    with ProcessPoolExecutor(12) as ex:
        for uid, st in ex.map(calibrate, uids):
            print(uid, st, flush=True)
