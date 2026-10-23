"""Detections -> per-rally ray observations (birdseye_v2.rally_obs) for every dataset, the 3D model's only input.

  drone  DroneData back view (BirdsCourt back camera, outputs/pose, the GT file's back-view TrackNet track,
         outputs/racket) + the evaluation targets: two-view shuttle GT, top-view player positions, labelled hits
  bfmd   BFMD broadcast rallies (calib_bc, pose1280, refined, racket)
  fb     FBBench clips (per-frame camtrack cameras, pose1280, refined, racket)
  fb20k  FineBadminton-20K training rallies (calib_bc, pose1280, refined when present)

Output: outputs/rally_obs/<set>/<uid>.npz (+ drone: <uid>_gt.json)
usage: python tools/build_obs.py drone|bfmd|fb|fb20k [uid ...]   (env WORKERS=8)"""
import csv, glob, json, os, sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
OUT = os.path.join(ROOT, "outputs", "rally_obs")
SET = sys.argv[1] if len(sys.argv) > 1 else "drone"
MAX_T = 2000  # frames
CLIPS = {"bfmd": ("bfmd", "clips"), "fb": ("fbbench", "hf_video_clips_qa"), "fb20k": ("fb20k", "clips")}


def video_info(path):
    import cv2
    c = cv2.VideoCapture(path)
    return int(c.get(cv2.CAP_PROP_FRAME_COUNT)), c.get(cv2.CAP_PROP_FPS) or 30.0


def uids():
    if SET == "drone":
        return sorted(f"{p.replace(os.sep, '/').split('/')[-2]}/{os.path.basename(p)[:-5]}"
                      for p in glob.glob(os.path.join(ROOT, "outputs", "gt", "*", "*.json")))
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(ROOT, "outputs", SET, "pose1280", "*.npz")))


def drone(mr):
    from birdseye_v2.court import Camera
    from birdseye_v2.players import load_pose, load_rackets
    from eval_players import gt_players
    m, r = mr.split("/")
    g = json.load(open(os.path.join(ROOT, "outputs", "gt", m, r + ".json")))
    cal = json.load(open(os.path.join(ROOT, "outputs", "calib_bc", m, r + ".json")))
    pp = os.path.join(ROOT, "outputs", "pose", m, r + ".npz")
    if not (os.path.exists(pp) and g.get("sync") and g.get("back_track") and cal.get("back")):
        return None
    cam, top = Camera.from_dict(cal["back"]), Camera.from_dict(cal["top"])
    poses = load_pose(pp)
    rp = os.path.join(ROOT, "outputs", "racket", m, r + ".npz")
    rk = load_rackets(rp) if os.path.exists(rp) else None
    T, fps = video_info(os.path.join(ROOT, "data", "videos", m, r, "(1).mp4"))
    T = min(T, len(poses))
    sh = (np.array(g["back_track"]["frames"], float), np.array(g["back_track"]["px"], float).reshape(-1, 2))
    off = g["sync"]["offset"]
    labels = [int(x["frame"]) + off for x in csv.DictReader(open(os.path.join(ROOT, "data", "datacsv", m, r, "1.csv")))
              if x["status"] not in ("0", "")]
    gp = {int(round(ft + off)): P.round(3).tolist() for ft, P in gt_players(m, r, top).items()}
    gt = {"shuttle": [[p["frame"] + off] + list(np.ravel(p["X"])) for p in g["points"]], "players": gp, "hits": labels,
          "cam": cal["back"]}
    return cam, poses, sh, rk, T, fps, gt


def frame_cams(base, uid, T, size):
    """Per-frame cameras from <base>/camtrack/<uid>.json when present, else the clip's calib_bc camera.
    (Not bfmd_stroke.frame_cams: importing that module loads Birdseye v1's training set, ~4 GB per worker.)"""
    from birdseye_v2 import camtrack
    from birdseye_v2.court import Camera
    p = os.path.join(base, "calib_bc", uid + ".json")
    c = json.load(open(p)).get("cam") if os.path.exists(p) else None
    cam = Camera.from_dict(c) if c else None
    tp = os.path.join(base, "camtrack", uid + ".json")
    if os.path.exists(tp):
        cams = camtrack.load(tp, cam.size if cam is not None else size)[:T]
        if any(x is not None for x in cams):
            return cams + [None] * (T - len(cams))
    return cam


def broadcast(uid):
    import cv2
    from birdseye_v2.players import load_pose, load_rackets
    base = os.path.join(ROOT, "outputs", SET)
    vp = os.path.join(ROOT, "data", *CLIPS[SET], uid + ".mp4")
    T, fps = video_info(vp)
    if T > MAX_T:  # multi-rally clips (FBBench long clips): not one rally, and memory-hungry
        return None
    c = cv2.VideoCapture(vp)
    cams = frame_cams(base, uid, T, (int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    if cams is None:
        return None
    poses = load_pose(os.path.join(base, "pose1280", uid + ".npz"))
    T = min(T, len(poses))
    sp = os.path.join(base, "refined", uid + ".json")
    sh = None
    if os.path.exists(sp):
        tr = json.load(open(sp))
        sh = (np.array(tr["frames"], float), np.array(tr["px"], float).reshape(-1, 2))
    rp = os.path.join(base, "racket", uid + ".npz")
    rk = load_rackets(rp) if os.path.exists(rp) else None
    return cams, poses, sh, rk, T, fps, None


def run(uid):
    from birdseye_v2.rally_obs import obs
    dst = os.path.join(OUT, SET, uid.replace("/", "__") + ".npz")
    if os.path.exists(dst):
        return None
    try:
        src = drone(uid) if SET == "drone" else broadcast(uid)
        if src is None:
            return f"skip {uid}"
        cams, poses, sh, rk, T, fps, gt = src
        o = obs(cams, poses, sh, rk, T=T, fps=fps)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst, **o)
        if gt is not None:
            json.dump(gt, open(dst[:-4] + "_gt.json", "w"))
        npl = int((o["kp_conf"].max(2) > 0).any(0).sum())
        return f"{uid} T={T} players={npl} shuttle={o['sh_vis'].mean():.2f} racket={(o['rk_conf'] > 0).any(1).mean():.2f}"
    except Exception as e:
        return f"FAILED {uid} {e!r}"


if __name__ == "__main__":
    us = sys.argv[2:] or uids()
    with Pool(int(os.environ.get("WORKERS", "8"))) as pool:
        for msg in pool.imap_unordered(run, us):
            if msg:
                print(msg, flush=True)
