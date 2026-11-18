"""Phase A (bev2_env, GPU): shared detectors + per-clip image stages.
  P   YOLOv8x-pose imgsz 1280, every frame (birdseye_v2.players.run_pose, as tools/fb_track.py POSE_IMGSZ=1280)
  R   YOLOv8x 'tennis racket' (class 38) imgsz 1920 conf 0.15, every frame (tools/run_racket.py loop)
  B1  YOLOv8x person boxes (class 0), ultralytics defaults (imgsz 640), every frame (Birdseye v1 2:Player detect_players_bfmd.py)
  S   TrackNetV3 (birdseye_v2.tracknet: resize + detect) + blur-centroid refinement (birdseye_v2.refine2d.refine_track)
  C2p BirdsCourt median-image prep (tools/fb_birdscourt.py prep)
  C1v VGGT-1B on every 10th frame (Birdseye v1 5.Evalutation/run_vggt_single_video.py; PNG write = decode, untimed)
Outputs go to /scratch/ciro/timing/root/outputs/cctv/* for the later phases."""
import glob, json, os, sys, time

os.environ["FB_SET"] = "cctv"
sys.path.insert(0, "/scratch/ciro/timing")
import tcommon as TC  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.backends.cudnn.allow_tf32 = False  # as tools/fb_track.py / run_pose.py / run_racket.py
torch.backends.cuda.matmul.allow_tf32 = True
sys.path.insert(0, TC.ROOT)
sys.path.insert(0, os.path.join(TC.ROOT, "tools"))
from ultralytics import YOLO  # noqa: E402
from birdseye_v2.players import run_pose, save_pose  # noqa: E402
from birdseye_v2.refine2d import refine_track  # noqa: E402
from birdseye_v2.tracknet import detect, load_model, read_small_frames, write_csv  # noqa: E402
import fb_birdscourt as BC  # noqa: E402

OUT = os.path.join(TC.ROOT, "outputs", "cctv")
W8 = os.path.join(TC.ROOT, "third_party", "weights")
VGGT_SNAP = glob.glob("/home/ciro/.cache/huggingface/hub/models--facebook--VGGT-1B/snapshots/*")[0]
VFR = os.path.join(TC.TIMING, "vggt_frames")
SKIP_VGGT = os.environ.get("SKIP_VGGT") == "1"

info = TC.hw_info()
print(info, flush=True)
load = {}
t0 = time.perf_counter(); tn = load_model(); TC.sync(); load["S_tracknet"] = time.perf_counter() - t0
t0 = time.perf_counter(); pose = YOLO(os.path.join(W8, "yolov8x-pose.pt")); load["P_yolov8x_pose"] = time.perf_counter() - t0
t0 = time.perf_counter(); rk = YOLO(os.path.join(W8, "yolov8x.pt")); load["R_yolov8x"] = time.perf_counter() - t0
t0 = time.perf_counter(); det = YOLO(os.path.join(W8, "yolov8x.pt")); load["B1_yolov8x"] = time.perf_counter() - t0
if not SKIP_VGGT:
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    vdtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    t0 = time.perf_counter(); vg = VGGT.from_pretrained(VGGT_SNAP).to("cuda").eval(); TC.sync(); load["C1_vggt1b"] = time.perf_counter() - t0
print("load", load, flush=True)

warm, test = TC.uids()
res = {"phase": "A", "hw": info, "load_s": load, "warmup_uid": warm, "clips": {}, "warmup": {}}
for k, uid in enumerate([warm] + test):
    src = TC.clip(uid)
    td = time.perf_counter(); TC.predecode([src]); tdec = time.perf_counter() - td
    n = TC.n_frames(src)
    T = TC.Timer()
    # P
    with T("P_pose1280"):
        kps = run_pose(src, pose, imgsz=1280)
    save_pose(kps, os.path.join(OUT, "pose1280", uid + ".npz"))
    # R
    with T("R_racket"):
        counts, boxes = [], []
        for r in rk.predict(src, imgsz=1920, conf=0.15, classes=[38], stream=True, verbose=False):
            b = r.boxes
            arr = np.c_[b.xyxy.cpu().numpy(), b.conf.cpu().numpy()] if len(b) else np.zeros((0, 5))
            counts.append(len(arr)); boxes.append(arr)
    os.makedirs(os.path.join(OUT, "racket"), exist_ok=True)
    np.savez_compressed(os.path.join(OUT, "racket", uid + ".npz"), counts=np.array(counts),
                        boxes=np.concatenate(boxes) if boxes else np.zeros((0, 5)))
    # B1 (v1 person boxes)
    with T("B1_v1_person_boxes"):
        pb = []
        for r in det.predict(src, classes=[0], stream=True, verbose=False):
            pb.append(r.boxes.xyxy.cpu().numpy())
    # S
    with T("S_tracknet_resize"):
        frames, full = read_small_frames(src)
    with T("S_tracknet"):
        rows = detect(tn, frames, full)
    write_csv(rows, os.path.join(OUT, "tracknet", uid + ".csv"))
    with T("S_refine"):
        vis = [r for r in rows if int(r[1]) == 1]
        idx = np.array([r[0] for r in vis], int)
        px, ok, _ = refine_track(src, (idx, np.array([[r[2], r[3]] for r in vis], float).reshape(-1, 2)))
    os.makedirs(os.path.join(OUT, "refined"), exist_ok=True)
    json.dump({"frames": idx[ok].tolist(), "px": np.round(px[ok], 2).tolist(), "n_det": int(len(idx))},
              open(os.path.join(OUT, "refined", uid + ".json"), "w"))
    del frames
    # C2 prep (BirdsCourt median image)
    ci = os.path.join(OUT, "court_img", uid + ".jpg")
    if os.path.exists(ci):
        os.remove(ci)
    with T("C2_birdscourt_prep"):
        BC.prep(uid)
    # C1 VGGT-1B focal seed (v1)
    if not SKIP_VGGT:
        d = os.path.join(VFR, uid)
        os.makedirs(d, exist_ok=True)
        fr = TC.CACHE[os.path.abspath(src)][0]
        names = []
        for j in range(0, len(fr), 10):  # ffmpeg select=not(mod(n,10)) -> PNG (decode/extraction: untimed)
            nm = os.path.join(d, f"frame_{j // 10 + 1:05d}.png")
            if not os.path.exists(nm):
                cv2.imwrite(nm, fr[j])
            names.append(nm)
        torch.cuda.reset_peak_memory_stats()
        try:
            with T("C1_vggt"):
                images = load_and_preprocess_images(names).to("cuda")
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=vdtype):
                        ib = images[None]
                        tok, ps_idx = vg.aggregator(ib)
                        pose_enc = vg.camera_head(tok)[-1]
                        extr, intr = pose_encoding_to_extri_intri(pose_enc, ib.shape[-2:])
                K = intr[0].float().cpu().numpy()
            w_img = images.shape[-1]
            fx = float(np.median(K[:, 0, 0]))
            hfov = float(np.degrees(2 * np.arctan((w_img / 2) / fx)))
            json.dump({"n_vggt_frames": len(names), "img_w": int(w_img), "fx_med": fx, "hfov_deg": hfov,
                       "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9},
                      open(os.path.join(OUT, "vggt_" + uid + ".json"), "w"))
            del images, tok
        except Exception as e:
            T.t.pop("C1_vggt", None)
            T.t["C1_vggt_error"] = repr(e)[:300]
        torch.cuda.empty_cache()
    TC.drop(src)
    rec = {"n_frames": n, "video_s": n / TC.FPS, "decode_s_untimed": tdec, **{k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in T.t.items()}}
    (res["warmup"] if k == 0 else res["clips"])[uid] = rec
    print(uid, json.dumps(rec), flush=True)
    TC.dump("phase_a", res)
