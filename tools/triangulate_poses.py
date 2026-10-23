"""Real 3D badminton poses from DroneData's two drones (dev matches BD12-14 only; the test matches are never
used): YOLOv8x-pose in the back view (outputs/pose) and the top view (outputs/pose_top), synced by the shuttle
offset of the GT file (back clock = top clock + offset), people matched by where they stand, every joint seen in
both views triangulated (birdseye_v2.twoview.triangulate_px). Seen from straight above, left and right limbs are
easily swapped, so the top skeleton is tried both ways and the one that reprojects better is kept.

Output: outputs/pose3d/<match>/<rally>.npz  {frames (N,), J (N,17,3) court frame m, NaN = rejected}
usage: python tools/triangulate_poses.py [BD12 BD13/Rally-4 ...]   (env WORKERS=8)"""
import glob, json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "outputs", "pose3d")
DEV = ("BD12", "BD13", "BD14")
FLIP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]  # COCO left <-> right
MIN_CONF = 0.5
MAX_ERR = 0.006  # reprojection error as a share of image width (~23 px at 4K)
MATCH_M = 1.0


def ground_top(cam, kp):
    """Where a person stands, from the top view: mid-hip ray cut at hip height (0.95 m)."""
    h = kp[[11, 12]]
    h = h[h[:, 2] > 0.3]
    if not len(h):
        return None
    return cam.pixel_to_ground(h[:, :2].mean(0)[None], z=0.95)[0][:2]


def run(mr):
    from scipy.optimize import linear_sum_assignment
    from birdseye_v2.court import Camera
    from birdseye_v2.players import foot_point, load_pose
    from birdseye_v2.twoview import triangulate_px
    m, r = mr.split("/")
    dst = os.path.join(OUT, m, r + ".npz")
    if os.path.exists(dst):
        return None
    try:
        g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
        cal = json.load(open(os.path.join(ROOT, "outputs", "calib_bc", m, r + ".json")))
        bp, tp = os.path.join(ROOT, "outputs", "pose", m, r + ".npz"), os.path.join(ROOT, "outputs", "pose_top", m, r + ".npz")
        if not (g.get("sync") and cal.get("back") and cal.get("top") and os.path.exists(bp) and os.path.exists(tp)):
            return f"skip {mr}"
        back, top = Camera.from_dict(cal["back"]), Camera.from_dict(cal["top"])
        off = g["sync"]["offset"]
        PB, PT = load_pose(bp), load_pose(tp)
        tol = MAX_ERR * max(back.size[0], top.size[0])
        frames, Js = [], []
        for fb in range(len(PB)):
            ft = int(round(fb - off))
            if not 0 <= ft < len(PT):
                continue
            pb = [(foot_point(back, k), k) for k in PB[fb]]
            pb = [(f, k) for f, k in pb if f is not None and -1 < f[0] < 7.1 and -1 < f[1] < 14.4]
            pt = [(ground_top(top, k), k) for k in PT[ft]]
            pt = [(f, k) for f, k in pt if f is not None]
            if not pb or not pt:
                continue
            C = np.array([[np.linalg.norm(a[0] - b[0]) for b in pt] for a in pb])
            ri, ci = linear_sum_assignment(C)
            for i, j in zip(ri, ci):
                if C[i, j] > MATCH_M:
                    continue
                kb, kt0 = pb[i][1], pt[j][1]
                best = None
                for kt in (kt0, kt0[FLIP]):
                    J = np.full((17, 3), np.nan)
                    errs = []
                    for q in range(17):
                        if kb[q, 2] < MIN_CONF or kt[q, 2] < MIN_CONF:
                            continue
                        X, e = triangulate_px([top, back], [kt[q, :2], kb[q, :2]])
                        if e.max() < tol:
                            J[q] = X
                        errs.append(min(e.max(), 3 * tol))
                    if errs and (best is None or np.mean(errs) < best[0]):
                        best = (np.mean(errs), J)
                if best is not None and np.isfinite(best[1][:, 0]).sum() >= 8:
                    frames.append(fb); Js.append(best[1])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst, frames=np.array(frames, int), J=np.array(Js, np.float32).reshape(-1, 17, 3))
        ok = np.isfinite(np.array(Js)[..., 0]).mean() if Js else 0
        return f"{mr} {len(Js)} poses, {ok:.2f} of joints kept"
    except Exception as e:
        return f"FAILED {mr} {e!r}"


if __name__ == "__main__":
    targets = sys.argv[1:] or list(DEV)
    mrs = []
    for t in targets:
        pat = t if "/" in t else t + "/*"
        mrs += [f"{p.replace(os.sep, '/').split('/')[-2]}/{os.path.basename(p)[:-4]}" for p in glob.glob(os.path.join(ROOT, "outputs", "pose_top", pat + ".npz"))]
    with Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for msg in pool.imap_unordered(run, sorted(mrs)):
            if msg:
                print(msg, flush=True)
