"""Small racket test (todo priority 2): COCO 'tennis racket' (class 38) from YOLOv8x on back-view
frames at annotated hits. Reports detection rate at hit frames and saves a contact sheet.
usage: python tools/racket_test.py BD12 [n_rallies]"""
import csv, glob, json, os, sys

import cv2
import numpy as np
import torch

torch.backends.cudnn.allow_tf32 = False
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ultralytics import YOLO

m = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
model = YOLO(os.path.join(ROOT, "third_party", "weights", "yolov8x.pt"))
tiles, stats = [], {"hits": 0, "frames_with_racket": 0, "rackets": 0}
for gj in sorted(glob.glob(os.path.join(ROOT, "outputs", "gt", m, "*.json")))[:n]:
    g = json.load(open(gj))
    if not g["sync"]:
        continue
    r = g["rally"]
    off = g["sync"]["offset"]
    hits = [int(x["frame"]) + off for x in csv.DictReader(open(os.path.join(ROOT, "data", "datacsv", m, r, "1.csv")))
            if x["status"] not in ("0", "")]
    cap = cv2.VideoCapture(os.path.join(ROOT, "data", "videos", m, r, "(1).mp4"))
    for h in hits:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(h)))
        ok, im = cap.read()
        if not ok:
            continue
        res = model.predict(im, imgsz=1920, conf=0.15, classes=[38], verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        stats["hits"] += 1
        stats["rackets"] += len(boxes)
        stats["frames_with_racket"] += int(len(boxes) > 0)
        vis = im.copy()
        for x0, y0, x1, y1 in boxes.astype(int):
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 6)
        cv2.putText(vis, f"{r} f{h:.0f} n={len(boxes)}", (40, 120), 0, 3, (255, 255, 255), 6)
        tiles.append(cv2.resize(vis, (640, 360)))
print(stats, "rate", round(stats["frames_with_racket"] / max(stats["hits"], 1), 3),
      "rackets/frame", round(stats["rackets"] / max(stats["hits"], 1), 2))
tiles = tiles[:24]
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(os.path.join(ROOT, "scratch", "racket_sheet.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
