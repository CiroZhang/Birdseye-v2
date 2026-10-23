"""YOLOv8x COCO 'tennis racket' (class 38) on every back-view frame ->
outputs/racket/<match>/<rally>.npz  (counts per frame + boxes xyxy,conf)
usage: python tools/run_racket.py BD12 BD13/Rally-4 ..."""
import glob, os, sys, time

import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = False
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from ultralytics import YOLO

VIDEOS = os.path.join(ROOT, "data", "videos")
OUT = os.path.join(ROOT, "outputs", "racket")
model = YOLO(os.path.join(ROOT, "third_party", "weights", "yolov8x.pt"))
for target in sys.argv[1:]:
    pattern = target if "/" in target else target + "/Rally-*"
    for rd in sorted(glob.glob(os.path.join(VIDEOS, pattern))):
        m, r = os.path.basename(os.path.dirname(rd)), os.path.basename(rd)
        dst = os.path.join(OUT, m, r + ".npz")
        if os.path.exists(dst):
            continue
        t0 = time.time()
        counts, boxes = [], []
        for res in model.predict(os.path.join(rd, "(1).mp4"), imgsz=1920, conf=0.15, classes=[38], stream=True, verbose=False):
            b = res.boxes
            arr = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy()] if len(b) else np.zeros((0, 5))
            counts.append(len(arr))
            boxes.append(arr)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        np.savez_compressed(dst, counts=np.array(counts), boxes=np.concatenate(boxes) if boxes else np.zeros((0, 5)))
        print(f"{m}/{r} {time.time() - t0:.1f}s", flush=True)
