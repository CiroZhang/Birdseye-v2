"""Full broadcast reconstruction for a few FineBadminton-20K rallies: VIMO players + 3D shuttle + rackets.

  shuttle  TrackNetV3 + blur-centroid refine (as tools/fb_track.py) -> v2's drag-physics RallyFit through the
           BirdsCourt camera, with the hits (frame, hitter) from the fine-tuned v1 model and hand anchors
  rackets  YOLOv8x COCO 'tennis racket' boxes; fixed 675 mm racket held in the VIMO hand, pointed at the box
           (lifted at the hand's depth), carried across short gaps, forearm direction otherwise (as tracks.py)
Cache: outputs/fb20k/{tracknet,refined,racket}/<uid>.*
usage: .venv_vimo/Scripts/python tools/fb_full_example.py uid ...   (called by tools/export_fb_viewer.py)"""
import json, os, sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2 import shuttle3d  # noqa: E402
from birdseye_v2.broadcast import FPS, SCALE, native_frames  # noqa: E402
from birdseye_v2.court import Camera  # noqa: E402
from birdseye_v2.pipeline import reconstruct_rally  # noqa: E402
from birdseye_v2.players import load_pose  # noqa: E402
from birdseye_v2.tracks import RACKET_HOLD, RACKET_LEN, SIGMA_RACKET, _fill, _smooth  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "fb20k")
CLIPS = os.path.join(ROOT, "data", "fb20k", "clips")
WRIST, ELBOW, HAND = {"L": 20, "R": 21}, {"L": 18, "R": 19}, {"L": 22, "R": 23}  # SMPL-24
HEAD_R = 0.48  # m, grip to racket-head centre (675 mm racket, 290 mm head, grip 80 mm from the butt)
TIP_R = RACKET_LEN - 0.08  # m, grip to racket tip


def detect_2d(uid):
    """TrackNet shuttle track + racket boxes, cached."""
    import torch
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = True
    src = os.path.join(CLIPS, uid + ".mp4")
    dst_r = os.path.join(OUT, "refined", uid + ".json")
    if not os.path.exists(dst_r):
        from birdseye_v2.refine2d import refine_track
        from birdseye_v2.tracknet import detect, load_model, read_small_frames, write_csv
        frames, full = read_small_frames(src)
        rows = detect(load_model(), frames, full)
        write_csv(rows, os.path.join(OUT, "tracknet", uid + ".csv"))
        vis = [r for r in rows if int(r[1]) == 1]
        idx = np.array([r[0] for r in vis], int)
        px, ok, _ = refine_track(src, (idx, np.array([[r[2], r[3]] for r in vis], float).reshape(-1, 2)))
        os.makedirs(os.path.dirname(dst_r), exist_ok=True)
        json.dump({"frames": idx[ok].tolist(), "px": np.round(px[ok], 2).tolist(), "n_det": int(len(idx))}, open(dst_r, "w"))
    dst_k = os.path.join(OUT, "racket", uid + ".npz")
    if not os.path.exists(dst_k):
        from ultralytics import YOLO
        model = YOLO(os.path.join(ROOT, "third_party", "weights", "yolov8x.pt"))
        counts, boxes = [], []
        for res in model.predict(src, imgsz=1280, conf=0.15, classes=[38], stream=True, verbose=False):
            b = res.boxes
            arr = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy()] if len(b) else np.zeros((0, 5))
            counts.append(len(arr)); boxes.append(arr)
        os.makedirs(os.path.dirname(dst_k), exist_ok=True)
        np.savez_compressed(dst_k, counts=np.array(counts), boxes=np.concatenate(boxes) if boxes else np.zeros((0, 5)))


def load_boxes(uid):
    d = np.load(os.path.join(OUT, "racket", uid + ".npz"))
    idx = np.r_[0, np.cumsum(d["counts"])]
    return [d["boxes"][idx[i]:idx[i + 1]] for i in range(len(d["counts"]))]


def shuttle(uid, cam, keep, hits):
    """Per native frame: 3D shuttle position (NaN outside the fitted rally) and the 2D detection (or None)."""
    T = len(keep)
    new_idx = {int(k): i for i, k in enumerate(keep)}
    tr = json.load(open(os.path.join(OUT, "refined", uid + ".json")))
    f, px = [], []
    for fi, p in zip(tr["frames"], tr["px"]):
        if fi in new_idx:
            f.append(new_idx[fi]); px.append(p)
    f, px = np.array(f, float), np.array(px, float).reshape(-1, 2)
    det = {int(a): b for a, b in zip(f, px)}
    poses_all = load_pose(os.path.join(OUT, "pose", uid + ".npz"))
    poses = []
    for k in keep:
        kp = poses_all[k].copy() if k < len(poses_all) else np.zeros((0, 17, 3), np.float32)
        kp[..., :2] *= SCALE
        poses.append(kp)
    cam3 = Camera(cam.f * SCALE, cam.rvec, cam.tvec, [int(s * SCALE) for s in cam.size], cam.dist)
    shuttle3d.FPS = FPS
    X = np.full((T, 3), np.nan)
    if len(f) >= 8:
        hits = sorted(h for h in hits if f[0] - 5 <= h <= f[-1])
        if not hits or hits[0] > f[0] + 4:
            hits = [float(f[0])] + hits
        rec = reconstruct_rally(cam3, f, px * SCALE, poses, hits=hits)
        fit = rec["fit"]
        lo, hi = int(fit.shots[0][0]), min(T - 1, int(np.ceil(fit.shots[-1][1])) + 15)
        ts = np.arange(lo, hi + 1, dtype=float)
        X[lo:hi + 1] = fit.position(ts, rest_until=hi)
    return X, det


def rackets(cam, J, boxes):
    """J: (T,24,3) VIMO world joints for one player. -> grip w (T,3), unit dir u (T,3), seen (T,) bool.
    cam: one Camera or one per frame (None = no court view: no racket box used)."""
    T = len(J)
    ok = np.isfinite(J[:, 0, 0])
    uv = np.full((T, 24, 2), np.nan)
    if isinstance(cam, list):
        cams = cam + [None] * (T - len(cam))
        ok &= np.array([c is not None for c in cams[:T]])
        for i in np.where(ok)[0]:
            uv[i] = cams[i].project(J[i])
    else:
        cams = [cam] * T
        uv[ok] = cam.project(J[ok].reshape(-1, 3)).reshape(-1, 24, 2)
    assign = []
    for i in range(min(T, len(boxes))):
        if not ok[i]:
            continue
        k = uv[i]
        x0, y0 = k.min(0); x1, y1 = k.max(0); w, h = x1 - x0, y1 - y0
        for bx0, by0, bx1, by1, conf in boxes[i]:
            c = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2])
            if conf >= 0.25 and x0 - 0.6 * w <= c[0] <= x1 + 0.6 * w and y0 - 0.7 * h <= c[1] <= y1 + 0.1 * h:
                d = {s: np.linalg.norm(c - k[WRIST[s]]) for s in "LR"}
                assign.append((i, (bx0, by0, bx1, by1), min(d, key=d.get)))
    votes = [s for _, _, s in assign]
    hand = max("LR", key=votes.count) if votes else "R"
    wr = J[:, HAND[hand]]
    fore = J[:, WRIST[hand]] - J[:, ELBOW[hand]]
    fore /= np.linalg.norm(fore, axis=1, keepdims=True) + 1e-9
    # no racket seen: forearm direction, but a hanging arm still holds the head up -> at least 0.2 rise
    h = np.linalg.norm(fore[:, :2], axis=1, keepdims=True) + 1e-9
    low = fore[:, 2] < 0.2
    fore[low] = np.c_[fore[low, :2] / h[low] * np.sqrt(1 - 0.2 ** 2), np.full(low.sum(), 0.2)]
    u_det = np.full((T, 3), np.nan)
    for i, b, s in assign:
        if s != hand:
            continue
        # The box spans the whole racket, so its centre is mid-shaft, not the head: the tip is the box corner
        # farthest from the hand, on the camera ray through it, TIP_R from the hand. Of the two ray-sphere
        # solutions keep the one closer to the forearm direction (keeping the higher one tipped rackets toward
        # or away from the camera); if the ray misses the sphere, its closest point. Checked from the top
        # drone (tools/eval_racket.py, 6 test rallies): tip 0.52 -> 0.27 m, direction 40 -> 14 deg.
        corners = np.array([[b[0], b[1]], [b[2], b[1]], [b[0], b[3]], [b[2], b[3]]])
        hp = cams[i].project(wr[i][None])[0]
        tip = corners[np.argmax(np.linalg.norm(corners - hp, axis=1))]
        cc, d = cams[i].rays(tip[None])
        cc = np.ravel(cc)[:3]
        d = d[0] / np.linalg.norm(d[0])
        tc = np.dot(wr[i] - cc, d)
        disc = TIP_R ** 2 - np.sum((cc + tc * d - wr[i]) ** 2)
        if disc <= 0 and np.sqrt(-disc + TIP_R ** 2) > 1.0:  # box ray passes > 1 m from the hand: not this racket
            continue
        tips = [cc + (tc + s * np.sqrt(disc)) * d for s in (-1, 1)] if disc > 0 else [cc + tc * d]
        vs = [(t - wr[i]) / (np.linalg.norm(t - wr[i]) + 1e-9) for t in tips]
        u_det[i] = max(vs, key=lambda v: v @ fore[i])
    seen = np.isfinite(u_det).all(1)
    u = _fill(np.arange(T), u_det, max_gap=2 * RACKET_HOLD)
    near = np.zeros(T, bool)
    for i in np.where(seen)[0]:
        near[max(0, i - RACKET_HOLD):i + RACKET_HOLD + 1] = True
    u[~near] = fore[~near]
    u = np.nan_to_num(u)
    u = _smooth(u, SIGMA_RACKET)
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    return wr, u, near, hand


def full(uid, cam, vimo, hits_native):
    """-> dict of per-frame arrays for the viewer."""
    detect_2d(uid)
    keep = vimo["frames"]
    T = len(vimo["top"])
    X, det = shuttle(uid, cam, keep[:T], hits_native)
    boxes_all = load_boxes(uid)
    boxes = [boxes_all[k] if k < len(boxes_all) else np.zeros((0, 5)) for k in keep[:T]]
    rk = {s: rackets(cam, vimo[s], boxes) for s in ("top", "bottom")}
    return {"X": X, "det": det, "rk": rk, "boxes": boxes}


if __name__ == "__main__":
    for u in sys.argv[1:]:
        detect_2d(u)
        print(u, "2D done", flush=True)
