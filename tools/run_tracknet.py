"""Run pretrained TrackNetV3 (lean GPU inference, birdseye_v2.tracknet) over DroneData videos.

Output: outputs/tracknet/<match>/<rally>/{top,back}.csv  (Frame,Visibility,X,Y,Conf; 4K px)
usage: python tools/run_tracknet.py BD12 BD13/Rally-4 ...   (no args = every downloaded match)"""
import glob, os, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.tracknet import detect, load_model, read_small_frames, write_csv

VIDEOS = os.path.join(ROOT, "data", "videos")
OUT = os.path.join(ROOT, "outputs", "tracknet")
VIEWS = {"1.mp4": "top", "(1).mp4": "back"}

model = load_model()
for target in sys.argv[1:] or sorted(os.listdir(VIDEOS)):
    pattern = target if "/" in target else target + "/Rally-*"
    for rally_dir in sorted(glob.glob(os.path.join(VIDEOS, pattern))):
        m, rally = os.path.basename(os.path.dirname(rally_dir)), os.path.basename(rally_dir)
        for vname, view in VIEWS.items():
            src = os.path.join(rally_dir, vname)
            dst = os.path.join(OUT, m, rally, f"{view}.csv")
            if not os.path.exists(src) or os.path.exists(dst):
                continue
            t0 = time.time()
            try:
                frames, full = read_small_frames(src)
                rows = detect(model, frames, full)
            except Exception as e:  # corrupt / partial download
                print("FAILED", src, e, flush=True)
                continue
            write_csv(rows, dst)
            print(f"{m}/{rally}/{view} {len(rows)}f {time.time() - t0:.1f}s", flush=True)
