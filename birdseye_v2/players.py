"""Back-view player detection (YOLOv8x-pose) and monocular 3D lifting of feet / wrists.

Feet: mid-ankle pixel ray-cast to the floor (z=0). Wrists: the wrist ray intersected with the
vertical plane through the player's foot point that faces the camera -- i.e. we assume the hand
is at the same depth as the feet, which is within ~0.5 m for a badminton swing."""
import os

import numpy as np

KP = {"l_sh": 5, "r_sh": 6, "l_wr": 9, "r_wr": 10, "l_hip": 11, "r_hip": 12, "l_ank": 15, "r_ank": 16}


def run_pose(video_path, model=None, imgsz=1280, conf=0.25):
    """-> list over frames of (N,17,3) keypoint arrays (x, y, score) in full-res pixels."""
    from ultralytics import YOLO

    model = model or YOLO("yolov8x-pose.pt")
    out = []
    for res in model.predict(video_path, imgsz=imgsz, conf=conf, stream=True, verbose=False, classes=[0]):
        k = res.keypoints
        if k is None or k.data is None or len(k.data) == 0:
            out.append(np.zeros((0, 17, 3), np.float32))
        else:
            out.append(k.data.cpu().numpy().astype(np.float32))
    return out


def save_pose(frames_kp, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    counts = np.array([len(k) for k in frames_kp])
    flat = np.concatenate([k for k in frames_kp if len(k)] or [np.zeros((0, 17, 3), np.float32)])
    np.savez_compressed(path, counts=counts, kp=flat)


def load_pose(path):
    d = np.load(path)
    idx = np.r_[0, np.cumsum(d["counts"])]
    return [d["kp"][idx[i]:idx[i + 1]] for i in range(len(d["counts"]))]


ANKLE_H = float(os.environ.get("ANKLE_H", "0.12"))  # m; chosen on dev BD12-14 (player error vs top view)


def foot_point(cam, kp, min_score=0.3):
    """Ground (x,y) of one person from the mid-ankle (fallback: lowest confident keypoint). The ankle
    keypoint is ANKLE_H above the floor: the ray is cut at that height, not at z=0 (cutting at the
    floor overshoots away from an elevated camera)."""
    ank = kp[[KP["l_ank"], KP["r_ank"]]]
    good = ank[ank[:, 2] > min_score]
    if len(good):
        uv = good[:, :2].mean(0)
    else:
        conf = kp[kp[:, 2] > min_score]
        if not len(conf):
            return None
        uv = conf[np.argmax(conf[:, 1]), :2]
    return cam.pixel_to_ground(uv[None], z=ANKLE_H)[0][:2]


def lift_at_depth(cam, uv, foot_xy):
    """3D point on the ray through uv whose distance along the camera's horizontal viewing
    direction equals that of foot_xy (vertical plane through the feet, facing the camera)."""
    c, d = cam.rays(np.asarray(uv, float)[None])
    d = d[0]
    fwd = cam.R.T @ np.array([0, 0, 1.0])
    fwd[2] = 0
    fwd /= np.linalg.norm(fwd)
    target = np.array([foot_xy[0], foot_xy[1], 0.0])
    denom = d @ fwd
    if abs(denom) < 1e-6:
        return None
    s = ((target - c) @ fwd) / denom
    return c + s * d


def players_3d(cam, kps, court_margin=1.5, min_score=0.3):
    """Per person in a frame: dict(foot=(x,y), wrists=[3D...]) for people standing on/near the
    court (rejects the adjacent court, spectators, line judges)."""
    from .court import L, W

    out = []
    for kp in kps:
        foot = foot_point(cam, kp, min_score)
        if foot is None or not (-court_margin < foot[0] < W + court_margin and -court_margin < foot[1] < L + court_margin):
            continue
        wr = []
        for j in (KP["l_wr"], KP["r_wr"]):
            if kp[j, 2] > min_score:
                p = lift_at_depth(cam, kp[j, :2], foot)
                if p is not None:
                    wr.append(p)
        out.append({"foot": foot, "wrists": wr, "kp": kp})
    return out


def load_rackets(path):
    d = np.load(path)
    idx = np.r_[0, np.cumsum(d["counts"])]
    return [d["boxes"][idx[i]:idx[i + 1]] for i in range(len(d["counts"]))]


def racket_anchor(cam, person, boxes, uv):
    """Racket-head 3D point for one person: the racket box (COCO 'tennis racket') inside the
    person's reach box that is closest to the shuttle pixel uv, its centre lifted at the
    person's foot depth. None if the person has no racket box."""
    kp = person["kp"]
    k = kp[kp[:, 2] > 0.3]
    if len(k) < 3 or not len(boxes):
        return None
    x0, y0 = k[:, :2].min(0)
    x1, y1 = k[:, :2].max(0)
    w, h = x1 - x0, y1 - y0
    x0, x1, y0, y1 = x0 - 0.6 * w, x1 + 0.6 * w, y0 - 0.7 * h, y1 + 0.1 * h
    best, bd = None, 1e9
    for bx0, by0, bx1, by1, _ in boxes:
        c = np.array([(bx0 + bx1) / 2, (by0 + by1) / 2])
        if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
            d = np.linalg.norm(c - uv)
            if d < bd:
                best, bd = c, d
    return None if best is None else lift_at_depth(cam, best, person["foot"])
