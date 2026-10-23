"""BirdsCourt calibration of DroneData's BACK camera (the pipeline's only input view).

DroneData courts are marked with yellow tape; BirdsCourt's MonoTrack line detector expects white paint. The
median back frame (players removed) is downscaled to 720p and every yellow-tape pixel is painted white; the
rest of the image is left as is. BirdsCourt's 22 points -> pinhole + 2-term radial fit (tools/fb_birdscourt.py)
-> scaled to the 4K video.

Output: outputs/calib_bc/<m>/<r>.json  {"back": BirdsCourt camera, "top": the existing top camera (ground truth
only, never seen by the pipeline), "stats": fit + agreement with the existing back calibration}.
If BirdsCourt fails, "back" falls back to the existing calibration and stats.fallback = true.
usage: python tools/drone_birdscourt.py [BD15 ...]"""
import glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(__file__))
from birdseye_v2.court import Camera, court_line_points, draw_court, median_frame  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "calib_bc")
IMG = os.path.join(OUT, "_img")
PTS = os.path.join(OUT, "_pts")


def prep(mr):
    m, r = mr
    dst = os.path.join(IMG, f"{m}_{r}.jpg")
    if os.path.exists(dst):
        return
    med = cv2.resize(median_frame(os.path.join(ROOT, "data", "videos", m, r, "(1).mp4")), (1280, 720), interpolation=cv2.INTER_AREA)
    tape = cv2.inRange(cv2.cvtColor(med, cv2.COLOR_BGR2HSV), (18, 60, 120), (40, 255, 255))
    tape = cv2.dilate(tape, np.ones((2, 2), np.uint8))
    os.makedirs(IMG, exist_ok=True)
    cv2.imwrite(dst, np.where(tape[..., None] > 0, 255, med).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])


def fit(mr):
    import fb_birdscourt as B
    m, r = mr
    old = json.load(open(os.path.join(ROOT, "outputs", "calib", m, r + ".json")))
    ref = Camera.from_dict(old["back"])
    p = os.path.join(PTS, f"{m}_{r}.json")
    pts = json.load(open(p)) if os.path.exists(p) else {"error": "no BirdsCourt output"}
    cam, st = (None, {"valid": False, "reason": pts["error"][:200]}) if "error" in pts else B.fit_camera(pts, (1280, 720), fit_dist=True)
    if cam is not None and st["valid"]:
        s = ref.size[0] / 1280.0
        cam = Camera(cam.f * s, cam.rvec, cam.tvec, ref.size, cam.dist)
        P = court_line_points()
        a, b = ref.project(P), cam.project(P)
        ok = (a[:, 0] >= 0) & (a[:, 0] < ref.size[0]) & (a[:, 1] >= 0) & (a[:, 1] < ref.size[1])
        d = np.linalg.norm(a[ok] - b[ok], axis=1)
        st.update(agree_median_px=round(float(np.median(d)), 1), agree_p90_px=round(float(np.percentile(d, 90)), 1),
                  centre_shift_m=round(float(np.linalg.norm(cam.center - ref.center)), 3), fallback=False)
        back = cam.to_dict()
        os.makedirs(os.path.join(OUT, m), exist_ok=True)
        im = cv2.imread(os.path.join(IMG, f"{m}_{r}.jpg"))
        cv2.imwrite(os.path.join(OUT, m, r + ".jpg"), draw_court(im, Camera(cam.f / s, cam.rvec, cam.tvec, (1280, 720), cam.dist), thick=1))
    else:
        st["fallback"] = True
        back = old["back"]
    os.makedirs(os.path.join(OUT, m), exist_ok=True)
    json.dump({"back": back, "top": old["top"], "stats": st, "source": "BirdsCourt"}, open(os.path.join(OUT, m, r + ".json"), "w"), indent=1)
    return mr, st


if __name__ == "__main__":
    import fb_birdscourt as B
    matches = sys.argv[1:] or sorted(os.listdir(os.path.join(ROOT, "outputs", "gt")))
    jobs = [(m, os.path.basename(p)[:-5]) for m in matches for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "gt", m, "Rally-*.json")))
            if os.path.exists(os.path.join(ROOT, "outputs", "calib", m, os.path.basename(p)[:-5] + ".json"))]
    with ProcessPoolExecutor(6) as ex:  # 15 4K frames per worker
        list(ex.map(prep, jobs))
    B.detect_images([os.path.join(IMG, f"{m}_{r}.jpg") for m, r in jobs], PTS)
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(fit, jobs))
    ok = [s for _, s in res if not s.get("fallback")]
    print(f"{len(res)} rallies, BirdsCourt used {len(ok)} ({100 * len(ok) / max(len(res), 1):.0f}%), "
          f"agreement with the existing calibration: median {np.median([s['agree_median_px'] for s in ok]):.1f} px at 4K, "
          f"camera shift median {np.median([s['centre_shift_m'] for s in ok]):.3f} m")
