"""Broadcast court calibration with BirdsCourt (github.com/CiroZhang/BirdsCourt): replaces tools/fb_calibrate.py's
line search, which often snaps to the wrong painted lines (court too small, players placed too near the camera).

  1. prep    per clip, the median of up to 15 court-view frames (players vanish) -> <OUT>/court_img/<uid>.jpg
  2. detect  BirdsCourt's 22 court-line points per image, run in WSL (Linux MonoTrack binary)
             -> <OUT>/court_pts/<uid>.json
  3. fit     pinhole camera (f, R, t; principal point at the centre) to the 22 points by least squares
             -> <OUT>/calib_bc/<uid>.json  {cam, stats: {valid, rms_px, n_pts, ...}} (fb_calibrate's format)
FB_SET=bench|train|bfmd as tools/fb_calibrate.py.
  2b. retry  clips whose median image failed: RETRY_N single frames, median of the solved frames' points
  4. courtfit clips BirdsCourt could not calibrate (side / net-post / corner views it was never trained on):
             the view-agnostic line-model fit (birdseye_v2.courtfit) on the median image, else on single frames
             (best one; stats.moving if they disagree) -> calib_bc/<uid>.json with source "courtfit"
usage: python tools/fb_birdscourt.py prep|detect|retry|fit|courtfit|all [uid ...]"""
import glob, json, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import COURT_CORNERS, Camera, draw_court, init_from_corners  # noqa: E402

FB_SET = os.environ.get("FB_SET", "bench")
CLIPS = os.path.join(ROOT, "data", *{"bench": ("fbbench", "hf_video_clips_qa"), "train": ("fb20k", "clips"),
                                     "bfmd": ("bfmd", "clips")}[FB_SET])  # bfmd: BFMD singles rally clips
OUT = os.path.join(ROOT, "outputs", {"bench": "fb", "train": "fb20k", "bfmd": "bfmd"}[FB_SET])
W, L, NET_Y, SS, LS, SIN = 6.1, 13.4, 6.7, 1.98, 0.76, 0.46
POINTS_3D = {  # BirdsCourt's names (its camera_calibration.COURT_POINTS_3D); same court frame as ours
    "P1_TL": (0, L), "P2_BL": (0, 0), "P3_BR": (W, 0), "P4_TR": (W, L),
    "P5_singlesTL": (SIN, L), "P6_singlesBL": (SIN, 0), "P7_singlesBR": (W - SIN, 0), "P8_singlesTR": (W - SIN, L),
    "P9_serviceFarL": (0, NET_Y + SS), "P10_serviceFarR": (W, NET_Y + SS),
    "P11_serviceNearL": (0, NET_Y - SS), "P12_serviceNearR": (W, NET_Y - SS),
    "P13_serviceFarCenter": (W / 2, NET_Y + SS), "P14_serviceNearCenter": (W / 2, NET_Y - SS),
    "P15_netL": (0, NET_Y), "P16_netR": (W, NET_Y),
    "P17_singlesFarL": (0, L - LS), "P18_singlesFarR": (W, L - LS),  # long service line for doubles (see BirdsCourt)
    "P19_singlesNearL": (0, LS), "P20_singlesNearR": (W, LS),
    "P21_baseFarCenter": (W / 2, L), "P22_baseNearCenter": (W / 2, 0),
}
MAX_RMS = 5.0  # px at 720p (4.0 rejected good red-court back views at 4.1-4.3)
DETECT_WORKERS = 4


def wsl_path(p):
    p = os.path.abspath(p).replace("\\", "/")
    return "/mnt/" + p[0].lower() + p[2:]


def frames(path):
    cap, i = cv2.VideoCapture(path), 0
    while True:
        ok, im = cap.read()
        if not ok:
            return
        yield i, im
        i += 1


def prep(uid):
    dst = os.path.join(OUT, "court_img", uid + ".jpg")
    if os.path.exists(dst):
        return
    path = os.path.join(CLIPS, uid + ".mp4")
    green = np.array([cv2.inRange(cv2.cvtColor(cv2.resize(im, (320, 180)), cv2.COLOR_BGR2HSV),
                                  (35, 60, 60), (95, 255, 255)).mean() / 255 for _, im in frames(path)])
    if not len(green):
        return
    good = np.where(green >= 0.85 * green.max())[0]
    sel = set(good[np.linspace(0, len(good) - 1, min(15, len(good))).astype(int)].tolist())
    med = np.median(np.stack([im for i, im in frames(path) if i in sel]), 0).astype(np.uint8)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, med, [cv2.IMWRITE_JPEG_QUALITY, 95])


def detect_images(imgs, out_dir):
    imgs = [p for p in imgs if os.path.exists(p) and not os.path.exists(os.path.join(out_dir, os.path.basename(p)[:-4] + ".json"))]
    if not imgs:
        return
    procs = []
    for k in range(DETECT_WORKERS):  # one WSL process per slice (MonoTrack + scorer are single-image, CPU)
        lst = os.path.join(os.path.dirname(imgs[0]), f"_todo{k}.txt")
        open(lst, "w", newline="\n").write("\n".join(wsl_path(p) for p in imgs[k::DETECT_WORKERS]) + "\n")
        procs.append(subprocess.Popen(["wsl", "-u", "root", "--", "/root/bc_venv/bin/python",
                                       wsl_path(os.path.join(ROOT, "tools", "birdscourt_wsl.py")),
                                       wsl_path(lst), wsl_path(out_dir)]))
    for pr in procs:
        pr.wait()


def detect(uids):
    detect_images([os.path.join(OUT, "court_img", u + ".jpg") for u in uids], os.path.join(OUT, "court_pts"))


RETRY_N = 10


def retry_prep(uid):
    """Failed median image (usually close-ups/replays mixed in): RETRY_N single frames spread over the clip."""
    path = os.path.join(CLIPS, uid + ".mp4")
    n = int(cv2.VideoCapture(path).get(cv2.CAP_PROP_FRAME_COUNT))
    want = set(np.linspace(0, max(n - 1, 0), RETRY_N + 2)[1:-1].astype(int).tolist())
    d = os.path.join(OUT, "court_img_retry")
    os.makedirs(d, exist_ok=True)
    for i, im in frames(path):
        if i in want:
            cv2.imwrite(os.path.join(d, f"{uid}_{i:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 95])


LR_PAIRS = (("P1_TL", "P4_TR"), ("P2_BL", "P3_BR"), ("P5_singlesTL", "P8_singlesTR"), ("P6_singlesBL", "P7_singlesBR"),
            ("P9_serviceFarL", "P10_serviceFarR"), ("P11_serviceNearL", "P12_serviceNearR"), ("P15_netL", "P16_netR"),
            ("P17_singlesFarL", "P18_singlesFarR"), ("P19_singlesNearL", "P20_singlesNearR"))


def unswap(pts):
    """BirdsCourt sometimes returns the court mirrored left/right (the court is symmetric, so the lines still
    match). Seen from behind the near baseline, the far-left corner is always left of the far-right one."""
    if "P1_TL" in pts and "P4_TR" in pts and pts["P1_TL"][0] > pts["P4_TR"][0]:
        sw = {n: n for n in pts}
        for a, b in LR_PAIRS:
            sw[a], sw[b] = b, a
        return {sw[n]: v for n, v in pts.items()}, True
    return pts, False


def retry_merge(uid):
    """Median of each court point over the single frames BirdsCourt solved -> court_pts/<uid>.json."""
    found, srcs = [], []
    for p in sorted(glob.glob(os.path.join(OUT, "court_pts_retry", uid + "_*.json"))):
        d = json.load(open(p))
        if "error" not in d and all(c in d for c in ("P1_TL", "P2_BL", "P3_BR", "P4_TR")):
            found.append(unswap(d)[0]); srcs.append(os.path.join(OUT, "court_img_retry", os.path.basename(p)[:-5] + ".jpg"))
    dst = os.path.join(OUT, "court_pts", uid + ".json")
    if not found:
        json.dump({"error": f"retry: no court in {RETRY_N} single frames"}, open(dst, "w"))
        return
    # frames from another camera shot that still "solve" are outliers: keep those near the corner median
    C = np.array([[f[c] for c in ("P1_TL", "P2_BL", "P3_BR", "P4_TR")] for f in found])
    ok = np.abs(C - np.median(C, 0)).max((1, 2)) < 25
    if ok.sum() < 2:  # no agreement: no single camera view dominates
        json.dump({"error": f"retry: {len(found)} frames solved but disagree"}, open(dst, "w"))
        return
    keep = [f for f, k in zip(found, ok) if k]
    used = [s for s, k in zip(srcs, ok) if k]
    pts = {k: np.median([f[k] for f in keep if k in f], 0).tolist() for k in POINTS_3D if any(k in f for f in keep)}
    pts["_retry"] = {"frames_solved": len(found), "frames_used": len(keep), "frames_tried": RETRY_N}
    json.dump(pts, open(dst, "w"))
    # the merged points are drawn on a frame that solved (the failed median image is kept beside it)
    img = os.path.join(OUT, "court_img", uid + ".jpg")
    if os.path.exists(img) and not os.path.exists(os.path.join(OUT, "court_img", uid + ".median_failed.jpg")):
        os.replace(img, os.path.join(OUT, "court_img", uid + ".median_failed.jpg"))
    cv2.imwrite(img, cv2.imread(used[len(used) // 2]))


def retry(uids):
    failed = [u for u in uids if os.path.exists(os.path.join(OUT, "court_pts", u + ".json"))
              and "error" in json.load(open(os.path.join(OUT, "court_pts", u + ".json")))
              and "retry" not in json.load(open(os.path.join(OUT, "court_pts", u + ".json")))["error"]]
    if not failed:
        return
    print(f"retry {len(failed)} failed clips with single frames", flush=True)
    for u in failed:
        os.rename(os.path.join(OUT, "court_pts", u + ".json"), os.path.join(OUT, "court_pts", u + ".median_failed.json"))
    with ProcessPoolExecutor(6) as ex:
        list(ex.map(retry_prep, failed))
    detect_images(sorted(glob.glob(os.path.join(OUT, "court_img_retry", "*.jpg"))), os.path.join(OUT, "court_pts_retry"))
    for u in failed:
        retry_merge(u)


def fit_camera(pts, size, fit_dist=False):
    """fit_dist: also 2 radial distortion terms (drone lenses), refined after the pinhole fit."""
    pts, swapped = unswap(pts)
    names = [n for n in POINTS_3D if n in pts]
    X = np.array([[*POINTS_3D[n], 0.0] for n in names])
    uv = np.array([pts[n] for n in names], float)
    corners = [pts.get(n) for n in ("P2_BL", "P3_BR", "P4_TR", "P1_TL")]
    if any(c is None for c in corners) or len(names) < 8:
        return None, {"valid": False, "n_pts": len(names), "reason": "missing corners"}
    best = None
    for f0 in (900, 1300, 1800, 2500, 3500):  # multi-start over focal length
        c0 = init_from_corners([tuple(corners[i]) for i in range(4)], size, float(f0))

        def res(p):
            return Camera(p[0], p[1:4], p[4:7], size).project(X).ravel() - uv.ravel()
        r = least_squares(res, np.r_[c0.f, c0.rvec, c0.tvec], loss="soft_l1", f_scale=3.0)
        if best is None or r.cost < best.cost:
            best = r
    cam = Camera(best.x[0], best.x[1:4], best.x[4:7], size)
    if fit_dist:
        def res_d(p):
            return Camera(p[0], p[1:4], p[4:7], size, p[7:9]).project(X).ravel() - uv.ravel()
        r = least_squares(res_d, np.r_[best.x, 0.0, 0.0], loss="soft_l1", f_scale=3.0)
        cam = Camera(r.x[0], r.x[1:4], r.x[4:7], size, r.x[7:9])
    e = np.linalg.norm(cam.project(X) - uv, axis=1)
    rms = float(np.sqrt(np.mean(e ** 2))) * 720.0 / size[1]
    c = cam.center
    sane = 300 < cam.f < 8000 and c[2] > 0.8 and c[1] < 0  # above the floor (low corner cameras sit ~1 m up), behind the near baseline
    return cam, {"valid": bool(rms < MAX_RMS and sane), "lr_swapped": swapped, "rms_px": round(rms, 2), "max_px": round(float(e.max()), 1),
                 "n_pts": len(names), "f": round(cam.f, 1), "center": [round(float(x), 2) for x in c]}


def fit(uid):
    p = os.path.join(OUT, "court_pts", uid + ".json")
    img = os.path.join(OUT, "court_img", uid + ".jpg")
    if not os.path.exists(p) or not os.path.exists(img):
        return uid, None
    pts = json.load(open(p))
    im = cv2.imread(img)
    size = (im.shape[1], im.shape[0])
    if "error" in pts:
        st, cam = {"valid": False, "reason": pts["error"][:200]}, None
    else:
        cam, st = fit_camera(pts, size)
        if "_retry" in pts:
            st["retry"] = pts["_retry"]
    os.makedirs(os.path.join(OUT, "calib_bc"), exist_ok=True)
    json.dump({"cam": cam.to_dict() if cam else None, "stats": st, "source": "BirdsCourt"},
              open(os.path.join(OUT, "calib_bc", uid + ".json"), "w"), indent=1)
    if cam is not None:
        ov = draw_court(im, cam, thick=1)
        for n, (x, y) in ((n, v) for n, v in pts.items() if n in POINTS_3D):
            cv2.circle(ov, (int(x), int(y)), 4, (0, 255, 0), 1)
        cv2.imwrite(os.path.join(OUT, "calib_bc", uid + ".jpg"), ov)
    return uid, st


CF_FRAMES = 12  # single frames tried when the median image does not calibrate


def courtfit(uid):
    """View-agnostic fallback for a clip without a valid BirdsCourt camera."""
    from birdseye_v2 import courtfit as CF
    dst = os.path.join(OUT, "calib_bc", uid + ".json")
    old = json.load(open(dst)) if os.path.exists(dst) else {"stats": {}}
    if old.get("stats", {}).get("valid"):
        return uid, None
    img = os.path.join(OUT, "court_img", uid + ".jpg")
    tried, sols = [], []
    if os.path.exists(img):
        im = cv2.imread(img)
        cam, st = CF.calibrate(im)
        tried.append(("median", st))
        if cam is not None:
            sols.append((st["score"], "median", cam, st, im))
    if not sols:  # moving camera (smeared median) or a median spoilt by ghosts: single frames
        cap = cv2.VideoCapture(os.path.join(CLIPS, uid + ".mp4"))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, max(n - 2, 0), CF_FRAMES).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, im = cap.read()
            if not ok:
                continue
            cam, st = CF.calibrate(im)
            if cam is not None:
                sols.append((st["score"], int(i), cam, st, im))
    birdscourt = {k: old.get("stats", {}).get(k) for k in ("valid", "reason", "rms_px", "center")}
    if not sols:
        st = {"valid": False, "reason": "courtfit: " + (tried[0][1].get("reason", "") if tried else "no image"), "birdscourt": birdscourt}
        json.dump({"cam": None, "stats": st, "source": "courtfit"}, open(dst, "w"), indent=1)
        return uid, st
    sc, frame, cam, st, im = max(sols, key=lambda x: x[0])
    centres = np.array([c.center for _, _, c, _, _ in sols])
    st = {**st, "valid": True, "frame": frame, "n_frames_solved": len(sols),
          "moving": bool(len(sols) > 1 and np.linalg.norm(centres - centres.mean(0), axis=1).max() > 1.0),
          "birdscourt": birdscourt}
    json.dump({"cam": cam.to_dict(), "stats": st, "source": "courtfit"}, open(dst, "w"), indent=1)
    cv2.imwrite(os.path.join(OUT, "calib_bc", uid + ".jpg"), draw_court(im, cam, thick=1))
    return uid, st


if __name__ == "__main__":
    step, uids = sys.argv[1], sys.argv[2:]
    if uids and uids[0].startswith("@"):
        uids = [l.strip() for l in open(uids[0][1:]) if l.strip()]
    uids = uids or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CLIPS, "*.mp4")))
    if step in ("prep", "all"):
        with ProcessPoolExecutor(6) as ex:  # streamed: ~15 full-res frames per worker
            list(ex.map(prep, uids, chunksize=4))
    if step in ("detect", "all"):
        detect(uids)
    if step in ("retry", "all"):
        retry(uids)
    if step in ("fit", "all"):
        with ProcessPoolExecutor(8) as ex:
            res = [r for r in ex.map(fit, uids) if r[1] is not None]
        ok = [s for _, s in res if s.get("valid")]
        print(f"fit {len(res)} clips, valid {len(ok)} ({100 * len(ok) / max(len(res), 1):.0f}%), "
              f"median rms {np.median([s['rms_px'] for s in ok]) if ok else float('nan'):.2f} px")
    if step in ("courtfit", "all"):
        with ProcessPoolExecutor(8) as ex:
            res = [r for r in ex.map(courtfit, uids) if r[1] is not None]
        ok = [s for _, s in res if s.get("valid")]
        views = {v: sum(s.get("view") == v for s in ok) for v in ("end", "side")}
        print(f"courtfit {len(res)} clips without a BirdsCourt camera: solved {len(ok)} {views}, "
              f"moving {sum(s.get('moving', False) for s in ok)}")
