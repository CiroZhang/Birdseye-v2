"""BFMD broadcasts keep one fixed main camera per match, so a rally whose own BirdsCourt fit failed (12%: other
lines snapped, f/distance traded off) borrows the camera of the nearest good rally of the same match.

  good   own fit valid and its centre within 1 m of the match median centre
  else   nearest good rally in time (a failed clip's own points are usually wrong lines, so they are not used
         to veto it); checked by line_score: share of projected court-line pixels on white paint
Rewrites outputs/bfmd/calib_bc/<uid>.json (originals kept in calib_bc_raw/), stats.source = own|neighbour:<uid>.
usage: python tools/bfmd_calib_fill.py"""
import glob, json, os, shutil, sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
os.environ["FB_SET"] = "bfmd"
from birdseye_v2.court import Camera  # noqa: E402
from fb_birdscourt import MAX_RMS, POINTS_3D, unswap  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "bfmd")
CAL, RAW = os.path.join(OUT, "calib_bc"), os.path.join(OUT, "calib_bc_raw")
MAX_DEV = 1.0  # m from the match median camera centre


def rms_on_points(cam, uid):
    p = os.path.join(OUT, "court_pts", uid + ".json")
    pts = json.load(open(p)) if os.path.exists(p) else {"error": ""}
    if "error" in pts:
        return None
    pts = unswap(pts)[0]
    names = [n for n in POINTS_3D if n in pts]
    X = np.array([[*POINTS_3D[n], 0.0] for n in names])
    e = np.linalg.norm(cam.project(X) - np.array([pts[n] for n in names], float), axis=1)
    return float(np.sqrt(np.mean(e ** 2))) * 720.0 / cam.size[1]


def line_score(cam, uid):
    """Share of the projected court lines (sampled every ~10 cm) that fall on white paint in the median image."""
    im = cv2.imread(os.path.join(OUT, "court_img", uid + ".jpg"))
    if im is None:
        return None
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    white = cv2.dilate(((hsv[..., 1] < 70) & (hsv[..., 2] > 170)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    W, L = 6.1, 13.4
    segs = [((0, y), (W, y)) for y in (0, 0.76, 6.7 - 1.98, 6.7 + 1.98, L - 0.76, L)] +            [((x, 0), (x, L)) for x in (0, 0.46, W - 0.46, W)] + [((W / 2, 0), (W / 2, 6.7 - 1.98)), ((W / 2, 6.7 + 1.98), (W / 2, L))]
    P = np.concatenate([np.linspace(a, b, int(np.hypot(b[0] - a[0], b[1] - a[1]) * 10) + 2) for a, b in map(lambda s: (np.array(s[0]), np.array(s[1])), segs)])
    uv = np.round(cam.project(np.c_[P, np.zeros(len(P))])).astype(int)
    h, w = white.shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return round(float(white[uv[ok, 1], uv[ok, 0]].mean()) if ok.any() else 0.0, 3)


def main():
    if not os.path.exists(RAW):
        shutil.copytree(CAL, RAW, ignore=shutil.ignore_patterns("*.jpg"))
    meta = os.path.join(ROOT, "data", "bfmd", "meta")
    n_own = n_nb = n_none = 0
    for m in open(os.path.join(meta, "all_matches.txt")).read().split():
        rallies = json.load(open(os.path.join(meta, m + ".rallies.json")))
        cal = {}
        for r in rallies:
            uid = f"{m}__g{r['game']}_r{r['rally']:03d}"
            p = os.path.join(RAW, uid + ".json")
            if os.path.exists(p):
                cal[uid] = (r["start"], json.load(open(p)))
        valid = {u: c for u, (_, c) in cal.items() if c["stats"].get("valid")}
        med = np.median([c["stats"]["center"] for c in valid.values()], 0)
        good = [u for u, c in valid.items() if np.linalg.norm(np.array(c["stats"]["center"]) - med) < MAX_DEV]
        for uid, (start, c) in cal.items():
            if uid in good:
                c["stats"]["source"] = "own"; n_own += 1
            else:
                cand = sorted(good, key=lambda u: abs(cal[u][0] - start))
                pick = cand[0] if cand else None
                if pick is not None:
                    err = rms_on_points(Camera.from_dict(cal[pick][1]["cam"]), uid)
                if pick is None:
                    c["stats"]["valid"] = False; c["stats"]["source"] = "none"; n_none += 1
                else:
                    own = c["stats"]
                    c = {"cam": cal[pick][1]["cam"], "source": "BirdsCourt",
                         "stats": {"valid": True, "source": "neighbour:" + pick, "rms_px": None if err is None else round(err, 2),
                                   "own": {k: own.get(k) for k in ("valid", "rms_px", "center", "f", "reason")}}}
                    n_nb += 1
            if c["cam"] is not None:
                c["stats"]["line_score"] = line_score(Camera.from_dict(c["cam"]), uid)
            json.dump(c, open(os.path.join(CAL, uid + ".json"), "w"), indent=1)
        print(f"{m[:50]}: {len(good)} own, {len(cal) - len(good)} filled", flush=True)
    print(f"own {n_own}, neighbour {n_nb}, none {n_none}")


if __name__ == "__main__":
    main()
