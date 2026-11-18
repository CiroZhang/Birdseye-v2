"""People on the court floor from every calibrated camera (TV, side, both player cameras), per TV frame.

Each pose detection (YOLOv8x-pose) gives a floor point: the mid-ankle pixel (fallback: lowest confident keypoint)
back-projected onto the plane z = ANKLE_H in that camera's court frame. Player-camera and side-camera frames are
mapped to TV time with the same sync as tools/body_mv.py. Used by the multi-camera player tracker.
usage: python tools/mv_people.py uid frame ...   (prints every camera's floor points at those TV frames)"""
import os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import body_mv as BM  # noqa: E402

ANKLE_H = 0.12
L_ANK, R_ANK = 15, 16
POSE = {"side": "cctv_side/pose_side", "p1": "cctv_p1/pose", "p2": "cctv_p2/pose"}   # side: full set from Killarney


def unproject(cam, uv, z=ANKLE_H):
    """pixel(s) (n,2) -> floor points (n,2) on the plane z for a camera dict (inverts the 2-term radial model)"""
    uv = np.atleast_2d(np.asarray(uv, float))
    xd = (uv - [cam["cx"], cam["cy"]]) / cam["f"]
    x = xd.copy()
    for _ in range(8):                                   # fixed-point undistortion
        r2 = (x ** 2).sum(1, keepdims=True)
        x = xd / (1 + cam["k1"] * r2 + cam["k2"] * r2 * r2)
    d = np.c_[x, np.ones(len(x))] @ cam["R"]             # ray directions in court coordinates
    C = np.asarray(cam["C"], float)
    t = (z - C[2]) / d[:, 2]
    P = C + t[:, None] * d
    P[t <= 0] = np.nan                                   # behind the camera / above the horizon
    return P[:, :2]


def rays(cam, uv):
    """pixel(s) (n,2) -> unit view directions (n,3) in court coordinates for a camera dict"""
    uv = np.atleast_2d(np.asarray(uv, float))
    xd = (uv - [cam["cx"], cam["cy"]]) / cam["f"]
    x = xd.copy()
    for _ in range(8):
        r2 = (x ** 2).sum(1, keepdims=True)
        x = xd / (1 + cam["k1"] * r2 + cam["k2"] * r2 * r2)
    d = np.c_[x, np.ones(len(x))] @ cam["R"]
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def foot_px(kp, min_score=0.3):
    ank = kp[[L_ANK, R_ANK]]
    g = ank[ank[:, 2] > min_score]
    if len(g):
        return g[:, :2].mean(0)
    g = kp[kp[:, 2] > min_score]
    return g[np.argmax(g[:, 1]), :2] if len(g) else None


def height_px(kp, min_score=0.3):
    g = kp[kp[:, 2] > min_score]
    return float(np.ptp(g[:, 1])) if len(g) >= 4 else 0.0


class People:
    """floor points of every camera's detections at a TV frame: people(f) -> {view: [(xy, kp, height_px)]}"""

    def __init__(s, uid):
        """cameras as in body_mv.cameras, but every view whose pose exists (TV pose1280, side pose_side or pose1280,
        player cameras pose)"""
        import json
        import cctv_teach as CTE
        import cctv_tri as CT
        import pcam_calib as PC
        from scipy.spatial.transform import Rotation
        from birdseye_v2.court import Camera
        s.views = {}
        tv = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")))["cam"])
        sc = CT.side_camera(uid)
        stat = [("tv", tv, 0, ["cctv/pose1280"])]
        if sc is not None:
            stat.append(("side", sc[0], sc[1], [POSE["side"], "cctv_side/pose1280"]))
        for v, cam, shift, dirs in stat:
            p = next((q for q in (os.path.join(ROOT, "outputs", d, uid + ".npz") for d in dirs) if os.path.exists(q)), None)
            if p:
                c = BM.static_cam(cam)
                s.views[v] = (lambda k, c=c: c, lambda k, sh=shift: k - sh, BM.load_kp(p))
        for v in ("p1", "p2"):
            cp = os.path.join(ROOT, "outputs", "cctv_" + v, "pcam", "calib", uid + ".npz")
            pp = os.path.join(ROOT, "outputs", POSE[v], uid + ".npz")
            if not (os.path.exists(cp) and os.path.exists(pp)):
                continue
            z = np.load(cp); R = Rotation.from_rotvec(z["rvec"]).as_matrix()
            cams = [dict(R=R[k], C=z["C"], f=z["f"][k], cx=PC.CX, cy=PC.CY, k1=float(z["k1"]), k2=0.0) for k in range(len(R))]
            s.views[v] = (lambda k, c=cams: c[k] if 0 <= k < len(c) else None, lambda k, sh=CTE.PCAM_SYNC[v]: k - sh, BM.load_kp(pp))

    def frame_of(s, v, f):
        """camera frame nearest to TV frame f"""
        _, tvt, K = s.views[v]
        k0 = int(round(f + (0 - tvt(0))))                 # tvt(k) = k - shift
        return k0 if 0 <= k0 < len(K) else None

    def people(s, f):
        out = {}
        for v, (camf, tvt, K) in s.views.items():
            k = s.frame_of(v, f)
            if k is None:
                continue
            cam = camf(k)
            if cam is None:
                continue
            pts = []
            for kp in K[k]:
                px = foot_px(kp)
                if px is None:
                    continue
                xy = unproject(cam, px)[0]
                if np.isfinite(xy).all():
                    pts.append((xy, kp, height_px(kp)))
            out[v] = pts
        return out


if __name__ == "__main__":
    P = People(sys.argv[1])
    print("views:", list(P.views))
    for f in map(int, sys.argv[2:]):
        for v, pts in P.people(f).items():
            print(f, v, [tuple(np.round(p[0], 1)) for p in pts])
