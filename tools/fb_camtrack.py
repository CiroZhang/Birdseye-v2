"""Per-frame cameras for FBBench / FineBadminton clips (birdseye_v2/camtrack.py): cuts, close-ups, other court
cameras (side, corner), pans and zooms. CPU only. Starts from the clip's calib_bc camera when it has one.

Output: outputs/<set>/camtrack/<uid>.json  {size, shots: [...], frames: [[f, rvec, tvec, dist] or null per frame]}
usage: python tools/fb_camtrack.py [uid ...]   (env FB_SET=bench|train|bfmd, WORKERS=12)"""
import glob, json, os, sys, time
from multiprocessing import Pool

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
FB_SET = os.environ.get("FB_SET", "bench")
CLIPS = os.path.join(ROOT, "data", *{"bench": ("fbbench", "hf_video_clips_qa"), "train": ("fb20k", "clips"),
                                     "bfmd": ("bfmd", "clips")}[FB_SET])
OUT = os.path.join(ROOT, "outputs", {"bench": "fb", "train": "fb20k", "bfmd": "bfmd"}[FB_SET])
POSE_DIR = os.environ.get("POSE_DIR", "pose1280")


def run(uid):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    from birdseye_v2 import camtrack as ct
    from birdseye_v2.court import Camera
    from birdseye_v2.players import load_pose
    dst = os.path.join(OUT, "camtrack", uid + ".json")
    if os.path.exists(dst):
        return None
    t0 = time.time()
    try:
        frames = ct.read_frames(os.path.join(CLIPS, uid + ".mp4"))
        cp = os.path.join(OUT, "calib_bc", uid + ".json")
        c = json.load(open(cp)).get("cam") if os.path.exists(cp) else None
        cams, info = ct.track(frames, Camera.from_dict(c) if c else None)
        pp = os.path.join(OUT, POSE_DIR, uid + ".npz")
        poses = None
        if os.path.exists(pp):
            sx = frames[0].shape[1] / 1280.0  # poses are in source pixels; frames are 1280 wide
            poses = [k.copy() for k in load_pose(pp)]
            if abs(sx - 1) > 1e-3:
                for k in poses:
                    k[..., :2] /= sx
        nf = ct.resolve_flips(frames, cams, info, poses)
        ct.save(dst, cams, info)
        have = float(np.mean([c is not None for c in cams]))
        return f"{uid} {len(frames)}f shots {len(info)} with-cam {have:.2f} views {[s.get('view') for s in info]} flipped {nf} {time.time() - t0:.0f}s"
    except Exception as e:
        return f"FAILED {uid} {e!r}"


if __name__ == "__main__":
    uids = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CLIPS, "*.mp4")))
    with Pool(int(os.environ.get("WORKERS", "12"))) as pool:
        for msg in pool.imap_unordered(run, uids):
            if msg:
                print(msg, flush=True)
