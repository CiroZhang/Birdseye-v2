"""Phase B (birdscourt_model_venv, CPU + reranker): court-line detection.
  C2  BirdsCourt 22-point detection on the median image (BirdsCourt Model/court_detection.detect: ffmpeg 3-frame avi
      -> net-free MonoTrack binary -> candidate pool -> trained reranker), as tools/birdscourt_wsl.py
  C1  Birdseye v1 / MonoTrack court detector: the original MonoTrack court-detection binary (24 points, reads the
      clip's middle frame itself), as Birdseye v1 1:Court (monotrack_bfmd_detections/*.out)"""
import json, os, subprocess, sys, time

sys.path.insert(0, "/scratch/ciro/timing")
BCM = "/scratch/ciro/BirdsCourt_repo/Model"
sys.path.insert(0, BCM)
MT_BIN = "/project/6101776/ciro/Badminton LLM/monotrack/court-detection/build/detect"
ROOT = "/scratch/ciro/timing/root"
OUT = os.path.join(ROOT, "outputs", "cctv")


def uids():
    test = json.load(open(os.path.join(OUT, "split.json")))["test"]
    n = int(os.environ.get("N_TEST", "20"))
    return test[n], test[:n]


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


import torch  # noqa: E402
import court_detection  # noqa: E402
import reranker  # noqa: E402

load = {}
t0 = time.perf_counter()
reranker._load(os.path.join(court_detection.WEIGHTS_DIR, "reranker_model.pt")); sync()
load["C2_birdscourt_reranker"] = time.perf_counter() - t0
warm, test = uids()
res = {"phase": "B", "load_s": load, "torch": torch.__version__, "reranker_device": str(reranker._load(
    os.path.join(court_detection.WEIGHTS_DIR, "reranker_model.pt"))[3]), "clips": {}, "warmup": {}}
os.makedirs(os.path.join(OUT, "court_pts"), exist_ok=True)
os.makedirs(os.path.join(OUT, "mt_court"), exist_ok=True)
for k, uid in enumerate([warm] + test):
    rec = {}
    img = os.path.join(OUT, "court_img", uid + ".jpg")
    sync(); t0 = time.perf_counter()
    try:
        pts = court_detection.detect(img)
        out = {k_: [float(v[0]), float(v[1])] for k_, v in pts.items()}
    except Exception as e:
        out = {"error": repr(e)}
    sync(); rec["C2_birdscourt_detect"] = time.perf_counter() - t0
    json.dump(out, open(os.path.join(OUT, "court_pts", uid + ".json"), "w"))
    rec["C2_n_pts"] = len([x for x in out if x != "error"])
    dst = os.path.join(OUT, "mt_court", uid + ".pts")
    if os.path.exists(dst):
        os.remove(dst)
    t0 = time.perf_counter()
    p = subprocess.run([MT_BIN, os.path.join(ROOT, "data", "cctv", "clips", uid + ".mp4"), dst], capture_output=True, text=True, timeout=300)
    rec["C1_monotrack_detect"] = time.perf_counter() - t0
    rec["C1_mt_ok"] = os.path.exists(dst)
    if not rec["C1_mt_ok"]:
        rec["C1_mt_err"] = (p.stdout[-200:] + p.stderr[-200:])
    (res["warmup"] if k == 0 else res["clips"])[uid] = rec
    print(uid, json.dumps(rec), flush=True)
os.makedirs("/scratch/ciro/timing/out", exist_ok=True)
json.dump(res, open("/scratch/ciro/timing/out/phase_b.json", "w"), indent=1)
