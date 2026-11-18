"""Per-clip scene for the QA experiments: what a QA model sees instead of (or with) the video.

For every clip of a set (fb = FBBench clips, fb20k = FineBadminton-20K training rallies), per native frame:
  t        time in the clip's video (s): native frame -> source frame (repeated frames dropped) / source fps
  2D       broadcast-view detections only, image coordinates scaled to 0-1000 (x right, y down):
           per side (top = far player, bottom = near player; the players tools/fb_vimo.py tracked) the mid-ankle,
           head (nose) and both wrists from YOLOv8-pose (pose1280); the TrackNet shuttle (refined)
  3D       the Birdseye student (RallyPhys, TAG) from the same detections, court frame in cm (x across 0-610,
           y along 0-1340 from the near baseline, z up): per side the pelvis, both wrists and the racket head; the
           shuttle; hits (time, side) from its contact head
Missing values are null. The window trimming and the text the model reads are made by the QA loader (tools/qa_train.py).

out  outputs/qa_scene/<set>/<uid>.json
usage: SET=fb|fb20k python tools/qa_scene.py   (student env as for evaluation, TAG=t3)"""
import glob, json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from birdseye_v2.court import NET_Y  # noqa: E402
from birdseye_v2.players import load_pose  # noqa: E402
from bfmd_export_2d import mid_ankle  # noqa: E402

SET = os.environ.get("SET", "fb")
TAG = os.environ.get("TAG", "t3")
OUTS = os.path.join(ROOT, "outputs", SET)
VID = {"fb": os.path.join(ROOT, "data", "fbbench", "hf_video_clips_qa"), "fb20k": os.path.join(ROOT, "data", "fb20k", "clips")}[SET]
DST = os.path.join(ROOT, "outputs", "qa_scene", SET)
WR = {"l": 9, "r": 10}  # COCO wrists
SMPL = {"pel": 0, "lw": 20, "rw": 21}


def video_info(uid):
    import cv2
    c = cv2.VideoCapture(os.path.join(VID, uid + ".mp4"))
    return c.get(cv2.CAP_PROP_FPS) or 25.0, int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(c.get(cv2.CAP_PROP_FRAME_COUNT))


def r(a, k=0):
    return None if a is None or not np.all(np.isfinite(a)) else [round(float(x), k) if k else int(round(float(x))) for x in a]


def scene_2d(uid, frames, d, W, H):
    """Per native frame and side: mid-ankle, head, wrists (0-1000); shuttle (0-1000)."""
    pp = os.path.join(OUTS, "pose1280", uid + ".npz")
    pose = load_pose(pp) if os.path.exists(pp) else []
    sx, sy = 1000.0 / W, 1000.0 / H
    out = {"top": [], "bottom": []}
    for side in out:
        ank = d[side + "_ankle_px"] if d is not None and side + "_ankle_px" in d.files else None
        for i, f in enumerate(frames):
            rec = None
            if ank is not None and f < len(pose) and len(pose[f]) and np.isfinite(ank[i]).all():
                dist = np.linalg.norm(mid_ankle(pose[f]) - ank[i], axis=1)
                if np.isfinite(dist).any() and np.nanmin(dist) < 80:
                    k = pose[f][int(np.nanargmin(dist))]
                    pt = lambda j: r(np.array([k[j, 0] * sx, k[j, 1] * sy])) if k[j, 2] > 0.3 else None
                    rec = {"feet": r(np.array([ank[i][0] * sx, ank[i][1] * sy])), "head": pt(0), "lw": pt(WR["l"]), "rw": pt(WR["r"])}
            out[side].append(rec)
    sh = [None] * len(frames)
    rp = os.path.join(OUTS, "refined", uid + ".json")
    if os.path.exists(rp):
        tr = json.load(open(rp))
        pos = {int(a): b for a, b in zip(tr["frames"], np.array(tr["px"], float).reshape(-1, 2))}
        for i, f in enumerate(frames):
            if int(f) in pos:
                sh[i] = r(pos[int(f)] * np.array([sx, sy]))
    out["shuttle"] = sh
    return out


def scene_3d(model, TR, uid, frames):
    """Per native frame i, the student at its raw video frame frames[i] (the student runs on every frame; FineBadminton
    clips repeat every 6th frame). Hits are given as native indices."""
    frames = np.asarray(frames, int)
    p = os.path.join(TR.OBS, SET, uid + ".npz")
    if not os.path.exists(p):
        return None
    f = TR.load_feats(p)
    o = TR.infer(model, f)
    nat = [i for i in range(len(frames)) if frames[i] < len(f["C"])]
    pres = f["present"] > 0
    sig = 1 / (1 + np.exp(-o["hit"])) * f["present"]
    hp = 1 - np.prod(1 - sig, axis=1)
    hit_frames = TR.hits_from(hp, nms=8)
    J = o["joints24"]
    hand = (np.nanmean(np.where(pres, o["hand_p"], np.nan), 0) > 0.5).astype(int)  # racket hand per slot
    u = o["u2"][:, np.arange(J.shape[1]), hand]
    u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
    w = o["w2"][:, np.arange(J.shape[1]), hand]
    head = w + 0.38 * u  # racket head centre, 0.38 m from the grip (as the contact loss)
    side_of = {}
    for side in ("top", "bottom"):
        slots = [q for q in range(J.shape[1]) if pres[:, q].sum() >= 8 and (np.median(f["root"][pres[:, q], q, 1]) > NET_Y) == (side == "top")]
        side_of[side] = sorted(slots, key=lambda q: -pres[:, q].sum())
    out = {"top": [], "bottom": [], "shuttle": [], "hits": []}
    for side in ("top", "bottom"):
        for i in nat:
            t = frames[i]
            q = next((q for q in side_of[side] if pres[t, q]), None)
            if q is None:
                out[side].append(None)
                continue
            cm = lambda a: r(np.asarray(a) * 100)
            out[side].append({"pel": cm(J[t, q, SMPL["pel"]]), "lw": cm(J[t, q, SMPL["lw"]]), "rw": cm(J[t, q, SMPL["rw"]]),
                              "racket": cm(head[t, q])})
    out["shuttle"] = [r(o["shuttle"][frames[i]] * 100) for i in nat]
    for h in hit_frames:
        i = int(np.searchsorted(frames, h, side="right")) - 1
        if 0 <= i < len(nat):
            q = int(np.argmax(sig[h]))
            side = "top" if np.median(f["root"][pres[:, q], q, 1]) > NET_Y else "bottom"
            out["hits"].append([i, side])
    return out


def main():
    import train_rallynet as TR
    model = TR.load_model(TAG)
    os.makedirs(DST, exist_ok=True)
    uids = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(VID, "*.mp4")))
    n_3d = n_mis = 0
    for k, uid in enumerate(uids):
        dp = os.path.join(DST, uid + ".json")
        if os.path.exists(dp):
            continue
        fps, W, H, nf = video_info(uid)
        vp = os.path.join(OUTS, "vimo_bc", uid + ".npz")
        d = np.load(vp) if os.path.exists(vp) else None
        frames = d["frames"] if d is not None else np.arange(nf)
        s2 = scene_2d(uid, frames, d, W, H) if d is not None else None
        s3 = scene_3d(model, TR, uid, frames)
        if s3 is not None:
            n_3d += 1
            if len(s3["shuttle"]) != len(frames):
                n_mis += 1
        json.dump({"uid": uid, "fps": fps, "size": [W, H], "t": [round(float(f) / fps, 3) for f in frames],
                   "d2": s2, "d3": s3}, open(dp, "w"))
        if k % 100 == 0:
            print(f"{k}/{len(uids)} {uid}: 3D {n_3d}, length mismatch {n_mis}", flush=True)
    print(f"done {SET}: {len(uids)} clips, with 3D {n_3d}, obs/vimo length mismatch {n_mis}", flush=True)


if __name__ == "__main__":
    main()
