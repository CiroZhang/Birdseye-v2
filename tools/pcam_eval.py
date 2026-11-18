"""Independent check of player-camera calibrations: the two-view (TV + side) triangulated shuttle projected into the
player camera vs that camera's own shuttle detections, and the TV players' feet vs the detected ankles.
usage: python tools/pcam_eval.py <view> [uid ...]"""
import json, os, sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import pcam_calib as PC
import cctv_tri as CT


def proj(z, k, X):
    R = Rotation.from_rotvec(z["rvec"][k]).as_matrix(); C = z["C"]
    Y = (X - C) @ R.T
    x, y = Y[:, 0] / Y[:, 2], Y[:, 1] / Y[:, 2]
    r2 = x * x + y * y
    d = 1 + float(z["k1"]) * r2
    depth = np.where(r2 > 1.6 * (np.hypot(PC.CX, PC.CY) / z["f"][k]) ** 2, -1.0, Y[:, 2])  # outside: distortion folds
    return np.c_[z["f"][k] * x * d + PC.CX, z["f"][k] * y * d + PC.CY], depth


def one(view, uid):
    p = os.path.join(ROOT, "outputs", "cctv_" + view, "pcam", "calib", uid + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p); n = len(z["f"]); sh = int(z["shift"])
    tri = {int(r[0]): np.array(r[1:]) for r in json.load(open(os.path.join(ROOT, "outputs", "cctv", "tri", uid + ".json")))["shuttle"]}
    rp = os.path.join(ROOT, "outputs", "cctv_" + view, "refined", uid + ".json")
    det = CT.track(rp) if os.path.exists(rp) else {}
    es = []
    for t, X in tri.items():
        k = t + sh
        if 0 <= k < n and k in det:
            uv, zz = proj(z, k, X[None])
            if zz[0] > 0.3:
                es.append(np.linalg.norm(uv[0] - det[k]) * zz[0] / z["f"][k])  # metres at the shuttle's depth
    ob = PC.player_obs(view, uid, sh, every=3)
    ef = []
    for k, ank, feet in ob:
        if k < n:
            if int(z["gap"][k]) > 3:  # only frames the court lines pin down
                continue
            uv, zz = proj(z, k, feet)
            m = zz > 0.3
            if m.any():
                d = np.linalg.norm(ank[:, None] - uv[m][None], axis=2)
                ef += (d.min(1) * zz[m][d.argmin(1)] / z["f"][k]).tolist()  # metres
    return {"uid": uid, "shuttle_n": len(es), "shuttle_med": float(np.median(es)) if es else None,
            "shuttle_ok": float(np.mean(np.array(es) < 0.15)) if es else None, "feet_med": float(np.median(ef)) if ef else None,
            "checked": int(len(z["anchored"])), "gap_med": float(np.median(z["gap"]))}


if __name__ == "__main__":
    view = sys.argv[1]
    uids = sys.argv[2:] or open(os.path.join(ROOT, "outputs", "cctv_pcam_dev_uids.txt")).read().split()
    R = [r for r in (one(view, u) for u in uids) if r]
    for r in R:
        print(r)
    s = [r["shuttle_med"] for r in R if r["shuttle_med"] is not None]; f = [r["feet_med"] for r in R if r["feet_med"] is not None]
    allok = [r["shuttle_ok"] for r in R if r["shuttle_ok"] is not None]
    print(f"SUMMARY {view}: {len(R)} rallies; shuttle: median of medians {np.median(s):.2f} m (rallies with detections {len(s)}), "
          f"share < 15 cm {np.mean(allok):.2f}; feet (line-pinned frames) median {np.median(f):.2f} m, rallies with feet > 0.4 m: {sum(x > 0.4 for x in f)}")
