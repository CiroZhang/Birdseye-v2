"""Calibrate top + back drone cameras for every downloaded DroneData rally by fitting the court
line model to the yellow line mask (birdseye_v2.court). Each rally is initialised from the
previous rally of the same match (drones hover, so the pose drifts only a little), the first rally
of a match from the BD12/Rally-1 reference.

Output: outputs/calib/<match>/<rally>.json  {top: cam, back: cam, *_stats}
        outputs/calib/<match>/<rally>_{top,back}.jpg   overlay thumbnails for QA
usage: python tools/calibrate_all.py BD12 BD13 ...  (no args = every downloaded match)"""
import glob, json, os, sys
import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import Camera, draw_court, fit_to_mask, init_from_corners, median_frame, yellow_mask

VIDEOS = os.path.join(ROOT, "data", "videos")
OUT = os.path.join(ROOT, "outputs", "calib")
SIZE = (3840, 2160)

# BD12/Rally-1 starting points (top: 4 hand-read corners from Birdseye; back: Birdseye's
# hand calibration) -- only used to seed the very first fit of each match
_R_BACK = np.array([[0.9999704, 0.00543918, -0.00544248],
                    [-0.00281185, -0.40008321, -0.91647451],
                    [-0.00716232, 0.91646268, -0.40005607]])
SEED_BACK = Camera(2472.5, cv2.Rodrigues(_R_BACK)[0], [-3.06615641, 1.90161457, 4.96874977], SIZE)
SEED_TOP = init_from_corners([(190, 286), (172, 1901), (3723, 1913), (3687, 296)], SIZE, 2472.5)


def fit_view(video, init, top):
    img = median_frame(video)
    mask = yellow_mask(img)
    c, st = fit_to_mask(init, mask, fit_f=False, fit_dist=False)
    # top view is near-fronto-parallel: focal length is weakly observable, so it is fit only
    # as a refinement after the pose has locked on
    c, st = fit_to_mask(c, mask, fit_f=True, fit_dist=True, truncs=(10, 5))
    return c, st, img


def rally_key(p):
    return int(os.path.basename(p).split("-")[1])


def main(matches):
    for m in matches:
        prev = {"top": SEED_TOP, "back": SEED_BACK}
        os.makedirs(os.path.join(OUT, m), exist_ok=True)
        for rd in sorted(glob.glob(os.path.join(VIDEOS, m, "Rally-*")), key=rally_key):
            rname = os.path.basename(rd)
            dst = os.path.join(OUT, m, rname + ".json")
            if os.path.exists(dst):
                d = json.load(open(dst))
                prev = {v: Camera.from_dict(d[v]) for v in ("top", "back")}
                continue
            res = {}
            for view, vid in (("top", "1.mp4"), ("back", "(1).mp4")):
                path = os.path.join(rd, vid)
                if not os.path.exists(path):
                    break
                cams = [fit_view(path, prev[view], view == "top")]
                if cams[0][1]["inlier_frac"] < 0.5:  # lost lock: retry from the reference seed
                    cams.append(fit_view(path, SEED_TOP if view == "top" else SEED_BACK, view == "top"))
                cam, st, img = max(cams, key=lambda c: c[1]["inlier_frac"])
                res[view] = cam.to_dict()
                res[view + "_stats"] = st
                cv2.imwrite(os.path.join(OUT, m, f"{rname}_{view}.jpg"),
                            cv2.resize(draw_court(img, cam, thick=3), None, fx=0.25, fy=0.25))
                if st["inlier_frac"] >= 0.5:
                    prev[view] = cam
            if len(res) == 4:
                json.dump(res, open(dst, "w"), indent=1)
                print(m, rname, "top %.2f back %.2f" % (res["top_stats"]["inlier_frac"],
                      res["back_stats"]["inlier_frac"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(os.listdir(VIDEOS)))
