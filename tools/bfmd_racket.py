"""YOLOv8x COCO 'tennis racket' (class 38) on every frame of the BFMD rally clips (as tools/run_racket.py for
DroneData) -> outputs/bfmd/<RACKET_DIR>/<uid>.npz (counts per frame + boxes xyxy,conf; load with
birdseye_v2.players.load_rackets). Clip frame i = 30 fps rally frame i.
usage: python tools/bfmd_racket.py [uid ...]   (env FB_SET=bfmd|bench, IMGSZ=1280, RACKET_DIR=racket; skips finished clips)"""
import glob, os, sys, time

import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = False  # lazy-JIT stall on the RTX 5080 (TF32 conv kernels JIT-compile for minutes on the RTX 5080)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from ultralytics import YOLO

FB_SET = os.environ.get("FB_SET", "bfmd")  # bench: FBBench clips (as tools/fb_track.py)
CLIPS = os.path.join(ROOT, "data", *{"bfmd": ("bfmd", "clips"), "bench": ("fbbench", "hf_video_clips_qa")}[FB_SET])
OUT = os.path.join(ROOT, "outputs", {"bfmd": "bfmd", "bench": "fb"}[FB_SET], os.environ.get("RACKET_DIR", "racket"))
IMGSZ = int(os.environ.get("IMGSZ", "1280"))
model = YOLO(os.path.join(ROOT, "third_party", "weights", "yolov8x.pt"))
uids = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CLIPS, "*.mp4")))
os.makedirs(OUT, exist_ok=True)
for uid in uids:
    dst = os.path.join(OUT, uid + ".npz")
    if os.path.exists(dst):
        continue
    t0 = time.time()
    counts, boxes = [], []
    try:
        for res in model.predict(os.path.join(CLIPS, uid + ".mp4"), imgsz=IMGSZ, conf=0.15, classes=[38], stream=True, verbose=False):
            b = res.boxes
            arr = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy()] if len(b) else np.zeros((0, 5))
            counts.append(len(arr))
            boxes.append(arr)
    except Exception as e:
        print("FAILED", uid, repr(e), flush=True)
        continue
    np.savez_compressed(dst, counts=np.array(counts), boxes=np.concatenate(boxes) if boxes else np.zeros((0, 5)))
    print(f"{uid} {len(counts)}f {np.mean(np.array(counts) > 0):.2f} {time.time() - t0:.1f}s", flush=True)
