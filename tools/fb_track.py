"""2D detections for FBBench broadcast clips: TrackNetV3 shuttle track (refined to the blur centroid)
and YOLOv8x-pose people.

Output: outputs/fb/tracknet/<uid>.csv, outputs/fb/refined/<uid>.json, outputs/fb/pose/<uid>.npz
usage: python tools/fb_track.py [uid ...]   (no args = every downloaded clip; skips finished ones)"""
import glob, json, os, sys, time

import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = False  # lazy-JIT stall on the RTX 5080 (TF32 conv kernels JIT-compile for minutes on the RTX 5080)
torch.backends.cuda.matmul.allow_tf32 = True
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ultralytics import YOLO
from birdseye_v2.players import run_pose, save_pose
from birdseye_v2.refine2d import refine_track
from birdseye_v2.tracknet import detect, load_model, read_small_frames, write_csv

# FB_SET=bench (FBBench clips), train (FineBadminton-20K rally clips, tools/fb20k_prep.py) or bfmd (BFMD singles rallies)
FB_SET = os.environ.get("FB_SET", "bench")
CLIPS = os.path.join(ROOT, "data", *{"bench": ("fbbench", "hf_video_clips_qa"), "train": ("fb20k", "clips"),
                                     "bfmd": ("bfmd", "clips")}[FB_SET])  # bfmd: BFMD singles rally clips
OUT = os.path.join(ROOT, "outputs", {"bench": "fb", "train": "fb20k", "bfmd": "bfmd"}[FB_SET])

tn = load_model()
POSE_IMGSZ = int(os.environ.get("POSE_IMGSZ", "768"))  # 1280 finds the far broadcast player in 94% of frames vs 72%
pose = YOLO(os.path.join(ROOT, "third_party", "weights", "yolov8x-pose.pt"))
uids = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CLIPS, "*.mp4")))
for uid in uids:
    src = os.path.join(CLIPS, uid + ".mp4")
    dst_t, dst_r, dst_p = (os.path.join(OUT, "tracknet", uid + ".csv"), os.path.join(OUT, "refined", uid + ".json"),
                           os.path.join(OUT, os.environ.get("POSE_DIR", "pose"), uid + ".npz"))
    pose_only = os.environ.get("POSE_ONLY") == "1"  # VIMO only needs player boxes
    shuttle_only = os.environ.get("SHUTTLE_ONLY") == "1"  # poses already made (e.g. POSE_ONLY at 1280)
    if os.path.exists(dst_r) if shuttle_only else os.path.exists(dst_p) and (pose_only or os.path.exists(dst_r)):
        continue
    t0 = time.time()
    try:
        if pose_only:
            save_pose(run_pose(src, pose, imgsz=POSE_IMGSZ), dst_p)
            print(f"{uid} pose {time.time() - t0:.1f}s", flush=True)
            continue
        frames, full = read_small_frames(src)
        rows = detect(tn, frames, full)
        write_csv(rows, dst_t)
        vis = [r for r in rows if int(r[1]) == 1]
        idx = np.array([r[0] for r in vis], int)
        px, ok, _ = refine_track(src, (idx, np.array([[r[2], r[3]] for r in vis], float).reshape(-1, 2)))
        os.makedirs(os.path.dirname(dst_r), exist_ok=True)
        json.dump({"frames": idx[ok].tolist(), "px": np.round(px[ok], 2).tolist(), "n_det": int(len(idx))}, open(dst_r, "w"))
        if not shuttle_only:
            save_pose(run_pose(src, pose, imgsz=768), dst_p)
    except Exception as e:
        print("FAILED", uid, repr(e), flush=True)
        continue
    print(f"{uid} {len(rows)}f {time.time() - t0:.1f}s", flush=True)
