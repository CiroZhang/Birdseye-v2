"""Runs inside WSL (BirdsCourt needs its Linux-built MonoTrack binary): court points for a list of images.
usage (WSL): /root/bc_venv/bin/python tools/birdscourt_wsl.py <list.txt> <out_dir>
  list.txt: one image path per line (WSL paths); writes <out_dir>/<image stem>.json {point_name: [x, y]}"""
import json, os, sys, traceback

BC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "BirdsCourt", "Model")
sys.path.insert(0, BC)
import court_detection  # noqa: E402

out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
for p in [l.strip() for l in open(sys.argv[1]) if l.strip()]:
    dst = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".json")
    if os.path.exists(dst):
        continue
    try:
        pts = court_detection.detect(p)
        json.dump({k: [float(v[0]), float(v[1])] for k, v in pts.items()}, open(dst, "w"))
        print("ok", os.path.basename(p), len(pts), flush=True)
    except Exception as e:
        json.dump({"error": repr(e)}, open(dst, "w"))
        print("FAILED", os.path.basename(p), repr(e)[:200], flush=True)
        traceback.print_exc()
