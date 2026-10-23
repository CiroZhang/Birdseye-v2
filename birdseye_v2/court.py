"""Court model + camera calibration for DroneData by fitting the badminton line model to the
yellow court-line mask (MonoTrack's detector is tuned for white lines and fails on this court).

World frame (same as Birdseye): x across the court width 0..6.1 (x=0 is the back-view camera's
LEFT doubles sideline), y along the length 0..13.4 (y=0 is the baseline nearest the back-view
camera), z up, metres.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

W, L = 6.1, 13.4
NET_Y = L / 2
POST_H = 1.55

# every painted line as a (p0, p1) ground segment
_xs = [0.0, 0.46, W - 0.46, W]
_ys_full = [0.0, 0.76, L / 2 - 1.98, L / 2 + 1.98, L - 0.76, L]
COURT_SEGMENTS = (
    [((x, 0.0), (x, L)) for x in _xs]
    + [((0.0, y), (W, y)) for y in _ys_full]
    + [((W / 2, 0.0), (W / 2, L / 2 - 1.98)), ((W / 2, L / 2 + 1.98), (W / 2, L))]
)
COURT_CORNERS = np.array([[0, 0, 0], [W, 0, 0], [W, L, 0], [0, L, 0]], float)


def court_line_points(step=0.05):
    """Dense 3D samples along every painted line (z=0)."""
    pts = []
    for (x0, y0), (x1, y1) in COURT_SEGMENTS:
        n = max(2, int(np.hypot(x1 - x0, y1 - y0) / step))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, np.zeros(n)], 1))
    return np.concatenate(pts)


def yellow_mask(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (20, 80, 120), (40, 255, 255))


def median_frame(video_path, n=15):
    """Median of n frames spread over the clip -- removes players/shuttle from the line mask."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, max(total - 2, 0), n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, im = cap.read()
        if ok:
            frames.append(im)
    return np.median(np.stack(frames), 0).astype(np.uint8)


class Camera:
    """Pinhole + 2-term radial distortion, principal point fixed at the image centre."""

    def __init__(self, f, rvec, tvec, size, dist=(0.0, 0.0)):
        self.f = float(f)
        self.rvec = np.asarray(rvec, float).reshape(3)
        self.tvec = np.asarray(tvec, float).reshape(3)
        self.size = tuple(int(s) for s in size)  # (w, h)
        self.dist = np.asarray(dist, float).reshape(2)

    @property
    def K(self):
        w, h = self.size
        return np.array([[self.f, 0, w / 2], [0, self.f, h / 2], [0, 0, 1.0]])

    @property
    def R(self):
        return cv2.Rodrigues(self.rvec)[0]

    @property
    def center(self):
        return -self.R.T @ self.tvec

    def project(self, X):
        X = np.ascontiguousarray(np.asarray(X, float).reshape(-1, 3))  # cv2 rejects strided views
        d = np.array([self.dist[0], self.dist[1], 0, 0])
        return cv2.projectPoints(X, self.rvec, self.tvec, self.K, d)[0].reshape(-1, 2)

    def undistort_px(self, uv):
        """Distorted pixels -> ideal pinhole pixels (same K)."""
        uv = np.asarray(uv, float).reshape(-1, 1, 2)
        d = np.array([self.dist[0], self.dist[1], 0, 0])
        return cv2.undistortPoints(uv, self.K, d, P=self.K).reshape(-1, 2)

    def rays(self, uv):
        """Pixels -> (camera centre, unit world-space directions)."""
        p = self.undistort_px(uv)
        d_cam = np.linalg.inv(self.K) @ np.c_[p, np.ones(len(p))].T
        d = (self.R.T @ d_cam).T
        return self.center, d / np.linalg.norm(d, axis=1, keepdims=True)

    def pixel_to_ground(self, uv, z=0.0):
        c, d = self.rays(uv)
        s = (z - c[2]) / d[:, 2]
        return c + s[:, None] * d

    def to_dict(self):
        return {"f": self.f, "rvec": self.rvec.tolist(), "tvec": self.tvec.tolist(),
                "size": list(self.size), "dist": self.dist.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["f"], d["rvec"], d["tvec"], d["size"], d.get("dist", (0, 0)))


def init_from_corners(corners_px, size, f):
    """PnP from the 4 outer court corners in COURT_CORNERS order."""
    w, h = size
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    ok, rvec, tvec = cv2.solvePnP(COURT_CORNERS, np.asarray(corners_px, float), K, None,
                                  flags=cv2.SOLVEPNP_IPPE)
    return Camera(f, rvec, tvec, size)


def fit_to_mask(cam, mask, fit_f=True, fit_dist=True, truncs=(30.0, 15.0, 8.0)):
    """Refine a camera so the projected line model lies on the yellow mask.

    Residual per model sample = truncated distance (px) from its projection to the nearest
    mask pixel. Truncation makes occluded / off-screen line samples harmless. Run coarse-to-
    fine by shrinking the truncation radius."""
    h, w = mask.shape
    dt = cv2.distanceTransform(255 - mask, cv2.DIST_L2, 5).astype(np.float32)
    pts = court_line_points()

    def unpack(p):
        i = 0
        f = cam.f
        if fit_f:
            f = p[0]; i = 1
        rvec, tvec = p[i:i + 3], p[i + 3:i + 6]
        dist = p[i + 6:i + 8] if fit_dist else cam.dist
        return Camera(f, rvec, tvec, cam.size, dist)

    def residuals(p, tr):
        c = unpack(p)
        uv = c.project(pts)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
        r = np.full(len(pts), tr, np.float32)
        if inside.any():
            m = cv2.remap(dt, uv[inside, 0].astype(np.float32).reshape(-1, 1),
                          uv[inside, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR)
            r[inside] = np.minimum(m.ravel(), tr)
        return r

    p = ([cam.f] if fit_f else []) + list(cam.rvec) + list(cam.tvec) + (list(cam.dist) if fit_dist else [])
    p = np.array(p, float)
    # small truncation from the start: a wide basin lets one painted line snap onto its
    # neighbour (e.g. doubles long-service line onto the short-service line)
    for tr in truncs:
        res = least_squares(residuals, p, args=(tr,), x_scale="jac", diff_step=1e-4, max_nfev=400)
        p = res.x
    final = unpack(p)
    r = residuals(p, truncs[-1])
    stats = {"median_px": float(np.median(r)), "inlier_frac": float((r < 5).mean()),
             "mean_inlier_px": float(r[r < 5].mean()) if (r < 5).any() else None}
    return final, stats


def draw_court(img, cam, color=(0, 0, 255), thick=2):
    out = img.copy()
    for (x0, y0), (x1, y1) in COURT_SEGMENTS:
        t = np.linspace(0, 1, 50)
        P = np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, np.zeros_like(t)], 1)
        uv = cam.project(P).astype(np.int32)
        cv2.polylines(out, [uv], False, color, thick)
    for x in (0.0, W):  # net posts + net top
        uv = cam.project([[x, NET_Y, 0], [x, NET_Y, POST_H]]).astype(np.int32)
        cv2.line(out, tuple(uv[0]), tuple(uv[1]), (255, 0, 255), thick)
    uv = cam.project([[0, NET_Y, POST_H], [W, NET_Y, POST_H]]).astype(np.int32)
    cv2.line(out, tuple(uv[0]), tuple(uv[1]), (255, 0, 255), thick)
    return out
