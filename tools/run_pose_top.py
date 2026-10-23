"""YOLOv8x-pose on every TOP-view clip (dev matches only, for the 3D pose prior) -> outputs/pose_top/<match>/<rally>.npz (birdseye_v2.players).
usage: python tools/run_pose_top.py BD12 BD13/Rally-4 ..."""
import glob, os, sys, time

import torch

torch.backends.cudnn.allow_tf32 = False  # same lazy-JIT stall as TrackNet on this GPU/torch build
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from ultralytics import YOLO
from birdseye_v2.players import run_pose, save_pose

VIDEOS = os.path.join(ROOT, "data", "videos")
OUT = os.path.join(ROOT, "outputs", "pose_top")
model = YOLO(os.path.join(ROOT, "third_party", "weights", "yolov8x-pose.pt"))
for target in sys.argv[1:] or sorted(os.listdir(VIDEOS)):
    pattern = target if "/" in target else target + "/Rally-*"
    for rd in sorted(glob.glob(os.path.join(VIDEOS, pattern))):
        m, r = os.path.basename(os.path.dirname(rd)), os.path.basename(rd)
        dst = os.path.join(OUT, m, r + ".npz")
        src = os.path.join(rd, "1.mp4")
        if os.path.exists(dst) or not os.path.exists(src):
            continue
        t0 = time.time()
        try:
            save_pose(run_pose(src, model), dst)
        except Exception as e:
            print("FAILED", src, e, flush=True)
            continue
        print(f"{m}/{r} {time.time() - t0:.1f}s", flush=True)
