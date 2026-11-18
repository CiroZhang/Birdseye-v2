"""Player-camera centres from hand-clicked court points (tools: the click page; data outputs/cctv/pcam_clicks/).
Per session x camera: one centre C and one radial distortion term shared by all clicked frames, a rotation and focal
length per frame; least squares on the clicked pixels (robust, so one mislabelled point cannot pull the fit).
-> outputs/cctv/pcam_<cam>_<session>.json {C, k1, frames: {id: {rvec, f, rms, per-point px}}}"""
import glob, json, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SIZE = (864, 486)
CX, CY = SIZE[0] / 2, SIZE[1] / 2
LM = json.load(open(os.path.join(ROOT, "outputs", "cctv", "pcam_clicks", "_frames.json")))["landmarks"]


def look_at(C, target):
    z = target - C; z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def project(C, rv, f, k1, X):
    Y = (X - C) @ Rotation.from_rotvec(rv).as_matrix().T
    z = np.maximum(Y[:, 2], 1e-3)
    x, y = Y[:, 0] / z, Y[:, 1] / z
    r2 = x * x + y * y
    d = 1 + k1 * r2
    return np.c_[f * x * d + CX, f * y * d + CY], Y[:, 2]


def solve(frames, C0):
    n = len(frames)

    def unpack(p):
        return p[:3], p[3], [(p[4 + 4 * i:7 + 4 * i], np.exp(p[7 + 4 * i])) for i in range(n)]

    def res(p):
        C, k1, fr = unpack(p)
        out = []
        for (rv, f), (X, uv) in zip(fr, frames):
            q, z = project(C, rv, f, k1, X)
            out.append(((q - uv) / 2.0).ravel()); out.append(np.minimum(z - 0.5, 0) * 100)
        return np.concatenate(out)

    p = [*C0, 0.0]
    for X, uv in frames:  # init: look at the clicked points' centroid, f from their spread
        R = look_at(np.asarray(C0, float), X.mean(0))
        p += [*Rotation.from_matrix(R).as_rotvec(), np.log(1000.0)]
    r = least_squares(res, np.array(p), loss="soft_l1", f_scale=1.5, max_nfev=3000, x_scale="jac")
    return r, unpack(r.x)


def main():
    docs = [json.load(open(p)) for p in glob.glob(os.path.join(ROOT, "outputs", "cctv", "pcam_clicks", "*.json")) if not os.path.basename(p).startswith("_")]
    for sess in ("sf", "d1026"):
        for cam in ("p1", "p2"):
            D = [d for d in docs if d["session"] == sess and d["cam"] == cam and not d.get("skip") and len(d["pts"]) >= 4]
            frames = [(np.array([LM[k] for k in d["pts"]], float), np.array(list(d["pts"].values()), float)) for d in D]
            best = None
            for C0 in [(x, y, z) for x in (-6, 0, 3, 6, 12) for y in (-5, 2, 6.7, 11, 18) for z in (1.0, 3.0) if not (-1 < x < 7 and -1 < y < 14.4)]:
                r, sol = solve(frames, C0)
                if best is None or r.cost < best[0].cost:
                    best = (r, sol)
            r, (C, k1, fr) = best
            out = {"C": np.round(C, 3).tolist(), "k1": round(float(k1), 4), "frames": {}}
            allres = []
            for d, (rv, f), (X, uv) in zip(D, fr, frames):
                q, _ = project(C, rv, f, k1, X)
                e = np.linalg.norm(q - uv, axis=1); allres += e.tolist()
                out["frames"][d["frame"]] = {"uid": d["uid"], "k": d["k"], "rvec": np.round(rv, 6).tolist(), "f": round(float(f), 2),
                                             "rms": round(float(np.sqrt(np.mean(e ** 2))), 2), "pts": {k: round(float(v), 1) for k, v in zip(d["pts"], e)}}
            json.dump(out, open(os.path.join(ROOT, "outputs", "cctv", f"pcam_{cam}_{sess}.json"), "w"), indent=1)
            print(sess, cam, len(D), "frames", "C", out["C"], "k1", out["k1"], "px median %.2f max %.1f" % (np.median(allres), np.max(allres)))
            for fid, v in out["frames"].items():
                bad = {k: e for k, e in v["pts"].items() if e > 4}
                print("   ", fid, "f", v["f"], "rms", v["rms"], ("outliers " + str(bad)) if bad else "")


if __name__ == "__main__":
    main()
