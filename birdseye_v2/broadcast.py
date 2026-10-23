"""Run the v2 back-view reconstruction on broadcast footage (FineBadminton / FBBench, 1280x720).

The broadcast camera sits behind and above the near baseline like DroneData's back drone, so the
same pipeline applies, with three adaptations:
  * pixel thresholds were tuned on 4K drone video where the court spans a similar fraction of the
    frame, so 720p coordinates are scaled x3 (camera, track, keypoints);
  * the footage is 25 fps. FBBench clips were padded to 30 fps by repeating every 6th frame; those
    repeats are detected and dropped so that frame k is time k/25 s again;
  * frames where the court is not in view (cuts, close-ups, replays) are dropped from the track;
  * with a camtrack file (birdseye_v2/camtrack.py: per-frame cameras, any view) the rally is solved in the main
    camera's image: pans and zooms of that camera are undone by moving each frame's pixels along their rays,
    frames from other cameras are dropped."""
import json, os

import cv2
import numpy as np

from . import camtrack, shuttle3d
from .court import Camera
from .pipeline import rally_summary, reconstruct_rally
from .players import load_pose

SCALE = 3.0
CAMTRACK = os.environ.get("CAMTRACK", "1") == "1"  # per-frame cameras from <out>/camtrack/<uid>.json when present
FPS = 25.0


def native_frames(video_path, cache=None):
    """Indices of the frames that are not repeats of the previous frame."""
    if cache and os.path.exists(cache):
        return np.array(json.load(open(cache)), int)
    cap = cv2.VideoCapture(video_path)
    prev, d = None, []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(im, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        d.append(np.inf if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    d = np.array(d)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps < 27:  # already native
        keep = np.arange(len(d))
    else:  # one repeat per 6 frames: in each run of 6, drop the frame most similar to its predecessor
        keep = []
        for s in range(0, len(d), 6):
            blk = np.arange(s, min(s + 6, len(d)))
            drop = blk[np.argmin(d[blk])] if len(blk) == 6 else None
            keep += [i for i in blk if i != drop]
        keep = np.array(keep, int)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(keep.tolist(), open(cache, "w"))
    return keep


def native_index(keep, frame):
    """Source frame number -> index in the native (repeat-free) sequence."""
    return int(np.clip(np.searchsorted(keep, frame), 0, len(keep) - 1))


def load_clip(root, uid, video_path, min_on_court=0.5, out="fb"):
    """Camera (x3), native-25fps track (frames renumbered 0..), poses and on-court flags."""
    fb = os.path.join(root, "outputs", out)
    keep = native_frames(video_path, os.path.join(fb, "native", uid + ".json"))
    new_idx = {int(k): i for i, k in enumerate(keep)}
    ctp = os.path.join(fb, "camtrack", uid + ".json")
    if CAMTRACK and os.path.exists(ctp):
        cams = camtrack.load(ctp)
        ref = camtrack.reference(cams)
        if ref is None:
            raise ValueError(f"{uid}: no court camera in any frame")
        move = lambda fi, uv: camtrack.transfer(cams[fi] if fi < len(cams) else None, ref, uv)
        on = np.array([move(fi, [[0.0, 0.0]]) is not None for fi in range(len(cams))], float)
        c = ref.to_dict()
    else:
        cal = json.load(open(os.path.join(fb, "calib", uid + ".json")))
        c = cal["cam"]
        on = np.array(cal["on_court"])
        move = lambda fi, uv: np.asarray(uv, float).reshape(-1, 2)
    cam = Camera(c["f"] * SCALE, c["rvec"], c["tvec"], [int(s * SCALE) for s in c["size"]], c.get("dist", (0, 0)))
    tr = json.load(open(os.path.join(fb, "refined", uid + ".json")))
    f, px = [], []
    for fi, p in zip(tr["frames"], tr["px"]):
        if fi in new_idx and fi < len(on) and on[fi] >= min_on_court:
            f.append(new_idx[fi]); px.append(move(fi, [p])[0])
    poses_all = load_pose(os.path.join(fb, "pose", uid + ".npz"))
    poses = []
    for k in keep:
        kp = poses_all[k].copy() if k < len(poses_all) else np.zeros((0, 17, 3), np.float32)
        if os.path.exists(ctp) and CAMTRACK:  # another camera's pixels: no transfer, no pose
            kp = kp if k < len(on) and on[k] else np.zeros((0, 17, 3), np.float32)
        if len(kp):
            kp[..., :2] = move(k, kp[..., :2].reshape(-1, 2)).reshape(kp.shape[:2] + (2,))
        kp[..., :2] *= SCALE
        poses.append(kp)
    on_native = np.array([on[k] if k < len(on) else 0 for k in keep])
    return cam, np.array(f, float), np.array(px, float).reshape(-1, 2) * SCALE, poses, on_native


def reconstruct_clip(root, uid, video_path, hit_model=None, hits=None, out="fb"):
    cam, f, px, poses, on = load_clip(root, uid, video_path, out=out)
    if len(f) < 8:
        return None
    shuttle3d.FPS = FPS  # physics time step for this footage
    rec = reconstruct_rally(cam, f, px, poses, hits=hits, hit_model=hit_model)
    rec.update(cam=cam, poses=poses, on_court=on, summary=rally_summary(rec, fps=FPS))
    return rec
