"""YOLOv8x COCO 'tennis racket' (class 38) on the TOP drone view, every STEP-th frame -- an independent check of
the back-view 3D rackets (tools/eval_racket.py); the top view is never used by the pipeline.
-> outputs/racket_top/<match>/<rally>.npz  (frames: top-clock frame idx, counts, boxes xyxy,conf in 4K px)
usage: python tools/top_racket.py BD15/Rally-39 ...   (env STEP=6, DEVICE=cpu, IMGSZ=1920)"""
import os, sys, time

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
from ultralytics import YOLO

STEP = int(os.environ.get("STEP", "6"))
DEVICE = os.environ.get("DEVICE", "cpu")
IMGSZ = int(os.environ.get("IMGSZ", "1920"))
model = YOLO(os.path.join(ROOT, "third_party", "weights", "yolov8x.pt"))
for mr in sys.argv[1:]:
    m, r = mr.split("/")
    dst = os.path.join(ROOT, "outputs", "racket_top", m, r + ".npz")
    if os.path.exists(dst):
        continue
    t0 = time.time()
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "videos", m, r, "1.mp4"))
    frames, counts, boxes, i = [], [], [], 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if i % STEP == 0:
            img = cap.retrieve()[1]
            b = model.predict(img, imgsz=IMGSZ, conf=0.15, classes=[38], device=DEVICE, verbose=False)[0].boxes
            arr = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy()] if len(b) else np.zeros((0, 5))
            frames.append(i); counts.append(len(arr)); boxes.append(arr)
        i += 1
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(dst, frames=np.array(frames), counts=np.array(counts),
                        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 5)))
    print(f"{mr} {len(frames)} frames, {np.mean(counts):.1f} boxes/frame, {time.time() - t0:.0f}s", flush=True)
