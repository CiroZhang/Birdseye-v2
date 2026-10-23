"""VIMO (TRAM) bodies for every player of a DroneData rally (back view only), for the player ablation
against the YOLO-pose lifting (outputs/ablation/players_yolo_ankle012/).

Players: tracked identities from birdseye_v2.tracks (doubles: up to 4), box = keypoint extent per frame.
VIMO with our calibrated back camera (frames and camera at half resolution, 1920x1080), joints to world,
then Birdseye v1's ground correction at ankle height (players.ANKLE_H). Cached-backbone fast path.
Output: outputs/drone_vimo/<match>/<rally>.npz  {joints: (P,T,24,3) world, NaN = missing}
usage: .venv_vimo/Scripts/python tools/drone_vimo.py BD12/Rally-1 ...   (or BD15 for a whole match)"""
import glob, json, os, shutil, sys, tempfile, time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import fb_vimo as FV  # noqa: E402  (TRAM import shims, fast_inference, ground_correct)
from fb_vimo import MAX_GAP, fill_box_gaps  # noqa: E402
from birdseye_v2.court import Camera  # noqa: E402
from birdseye_v2.players import load_pose  # noqa: E402
from birdseye_v2.tracks import smooth_track, track_players  # noqa: E402

ROOT = FV.ROOT
HALF = 0.5
DRONE_CALIB = os.environ.get("DRONE_CALIB", "calib")  # "calib_bc" = BirdsCourt (tools/drone_birdscourt.py)
DRONE_VIMO = os.environ.get("DRONE_VIMO", "drone_vimo")
PATCH = os.environ.get("PATCH") == "1"  # redo only the players whose gap filling adds frames, keep the rest


def redo(tr, old_raw, T):
    """PATCH: does gap filling give this track >10% more VIMO frames than the old run?"""
    kp = tr["kp"][:T]
    valid = np.isfinite(kp[:, 0, 0]) & ((kp[:, :, 2] > 0.3).sum(1) >= 5)
    _, v = fill_box_gaps(np.zeros((len(valid), 4)), valid.copy())
    return v.sum() >= 16 and np.isfinite(old_raw[:T, 0, 0]).sum() < 0.9 * v.sum()


def run(model, m, r):
    dst = os.path.join(ROOT, "outputs", DRONE_VIMO, m, r + ".npz")
    old = None
    if os.path.exists(dst):
        if not PATCH:
            return
        old = dict(np.load(dst))
        if old.get("gapfill") is not None:
            return
    t0 = time.time()
    cam = Camera.from_dict(json.load(open(os.path.join(ROOT, "outputs", DRONE_CALIB, m, r + ".json")))["back"])
    poses = load_pose(os.path.join(ROOT, "outputs", "pose", m, r + ".npz"))
    T = len(poses)
    tracks = [smooth_track(cam, t) for t in track_players(cam, poses, np.arange(T))]
    cam_h = Camera(cam.f * HALF, cam.rvec, cam.tvec, [int(s * HALF) for s in cam.size], cam.dist)
    if old is not None and not any(redo(tr, old["raw"][k], T) for k, tr in enumerate(tracks) if k < len(old["raw"]))             and len(tracks) <= len(old["raw"]):
        np.savez_compressed(dst, **old, gapfill=np.int8(MAX_GAP))  # nothing to add: skip the frame extraction
        return
    tmp = tempfile.mkdtemp(prefix="dvimo_")
    try:
        cap, files = cv2.VideoCapture(os.path.join(ROOT, "data", "videos", m, r, "(1).mp4")), []
        while len(files) < T:
            ok, im = cap.read()
            if not ok:
                break
            fn = os.path.join(tmp, f"{len(files):06d}.jpg")
            cv2.imwrite(fn, cv2.resize(im, cam_h.size, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
            files.append(fn)
        T = len(files)
        out = np.full((len(tracks), T, 24, 3), np.nan, np.float32)
        raw = np.full_like(out, np.nan)
        ank = np.full((len(tracks), T, 2), np.nan, np.float32)
        R = cam.R
        for k, tr in enumerate(tracks):
            kp = tr["kp"][:T]
            valid = np.isfinite(kp[:, 0, 0])
            boxes = np.zeros((T, 4), np.float32)
            ankle = np.full((T, 2), np.nan)
            for i in np.where(valid)[0]:
                q = kp[i][kp[i][:, 2] > 0.3]
                if len(q) < 5:
                    valid[i] = False
                    continue
                x1, y1 = q[:, :2].min(0); x2, y2 = q[:, :2].max(0)
                pad = 0.1 * max(x2 - x1, y2 - y1)
                boxes[i] = np.array([x1 - pad, y1 - pad, x2 + pad, y2 + pad]) * HALF
                a = kp[i][[15, 16]]
                a = a[a[:, 2] > 0.3]
                if len(a):
                    ankle[i] = a[:, :2].mean(0)  # full-res px for the ground correction (full-res camera)
            boxes, valid = fill_box_gaps(boxes, valid)
            if valid.sum() < 16:
                continue
            if old is not None and k < len(old["raw"]) and np.isfinite(old["raw"][k, :T, 0, 0]).sum() >= 0.9 * valid.sum():
                raw[k], ank[k], out[k] = old["raw"][k, :T], old["ankle"][k, :T], old["joints"][k, :T]  # gaps add little
                continue
            res = FV.fast_inference(model, files, boxes, valid, float(cam_h.f), np.array(cam_h.size, np.float32) / 2)
            if res is None:
                continue
            so = model.smpl.query({"pred_rotmat": res["pred_rotmat"].float().cuda(), "pred_shape": res["pred_shape"].float().cuda()},
                                  default_smpl=True)
            j_cam = (so.joints[:, :24].float().cpu() + res["pred_trans"].float()).numpy()
            j_w = (R.T @ (j_cam - cam.tvec).transpose(0, 2, 1)).transpose(0, 2, 1)
            J = np.full((T, 24, 3), np.nan)
            J[res["frame"].numpy().astype(int)] = j_w
            raw[k], ank[k] = J, ankle
            out[k] = FV.ground_correct(cam, J, ankle)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst, joints=out, raw=raw, ankle=ank, gapfill=np.int8(MAX_GAP))
        print(f"{m}/{r} {T}f players {len(tracks)} cov {np.isfinite(out[:, :, 0, 0]).mean():.2f} {time.time() - t0:.0f}s", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    os.chdir(FV.TRAM)
    from lib.models import get_hmr_vimo
    model = get_hmr_vimo(checkpoint=os.path.join(FV.TRAM, "data", "pretrain", "vimo_checkpoint.pth.tar"))
    jobs = []
    for a in sys.argv[1:]:
        if "/" in a:
            jobs.append(tuple(a.split("/")))
        else:
            jobs += [(a, os.path.basename(p)[:-5]) for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "gt", a, "*.json")))]
    for m, r in jobs:
        try:
            run(model, m, r)
        except Exception as e:
            print("FAILED", m, r, repr(e), flush=True)
