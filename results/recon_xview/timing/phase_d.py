"""Phase D (TRAM env, GPU): Birdseye v1 player stages.
  V   VIMO for up to 4 tracked players per frame (tools/fb_vimo.py main, VIMO_N4=1 CAMTRACK=0, calib_bc camera,
      pose1280 boxes) -- per-clip wall time of main()'s loop body minus the ground correction; includes its
      per-frame JPEG dump (TRAM's TrackDataset reads image files), box selection, ViT-H backbone + temporal heads,
      SMPL joints
  G   v1 ground correction (fb_vimo.ground_correct)
  H   Birdseye v1 hit / stroke transformer (birdseye_v2.v1action: HitSpotterTransformer + mirror TTA) on the
      top0 / bottom0 bodies
Clip decoding happens inside the native_frames hook and is excluded from the clip's time."""
import json, os, sys, time

os.environ.update(FB_SET="cctv", VIMO_N4="1", VIMO_DIR="vimo_n4", POSE_DIR="pose1280", CAMTRACK="0")
sys.path.insert(0, "/scratch/ciro/timing")
import tcommon as TC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(TC.ROOT, "tools"))
import fb_vimo as FV  # noqa: E402

info = TC.hw_info()
warm, test = TC.uids()
order = [warm] + test
OUT = os.path.join(TC.ROOT, "outputs", "cctv")
for u in order:
    p = os.path.join(OUT, "vimo_n4", u + ".npz")
    if os.path.exists(p):
        os.remove(p)

marks = []  # (uid, t_start_after_decode, t_before_decode_of_next)
acc = {}  # uid -> {stage: s}
cur = {"uid": None}


def add(name, dt):
    d = acc.setdefault(cur["uid"], {})
    d[name] = d.get(name, 0.0) + dt


_nf = FV.native_frames


def native_frames_hook(video, cache=None):
    TC.sync()
    t_a = time.perf_counter()
    if marks:
        marks[-1][2] = t_a
    for p in list(TC.CACHE):
        TC.drop(p)
    TC.predecode([video])  # untimed
    uid = os.path.basename(video)[:-4]
    cur["uid"] = uid
    acc.setdefault(uid, {})["decode_s_untimed"] = time.perf_counter() - t_a
    t_b = time.perf_counter()
    marks.append([uid, t_b, None])
    return _nf(video, cache)


def timed(fn, name, gpu=False):
    def w(*a, **k):
        if gpu:
            TC.sync()
        t0 = time.perf_counter()
        r = fn(*a, **k)
        if gpu:
            TC.sync()
        add(name, time.perf_counter() - t0)
        return r
    return w


FV.native_frames = native_frames_hook
FV.ground_correct = timed(FV.ground_correct, "G_ground_correct")
FV.fast_chunk = timed(FV.fast_chunk, "V_sub_vit_chunks", gpu=True)
FV.side_boxes4 = timed(FV.side_boxes4, "V_sub_box_select")
FV.cv2.imwrite = timed(FV.cv2.imwrite, "V_sub_jpeg_dump")

sys.argv = ["fb_vimo.py"] + order
t_main = time.perf_counter()
FV.main()
TC.sync()
t_end = time.perf_counter()
marks[-1][2] = t_end
load = {"V_vimo_model": marks[0][1] - t_main - acc[order[0]]["decode_s_untimed"]}

clips = {}
for uid, a, b in marks:
    d = acc.get(uid, {})
    tot = b - a
    clips[uid] = {"V_total_wall": tot, "V_vimo": tot - d.get("G_ground_correct", 0.0), **d}

# Birdseye v1 hit transformer
sys.path.insert(0, TC.ROOT)
from birdseye_v2 import v1action  # noqa: E402
t0 = time.perf_counter(); hm = v1action.load(); TC.sync(); load["H_v1_hit_transformer"] = time.perf_counter() - t0
for uid in order:
    p = os.path.join(OUT, "vimo_n4", uid + ".npz")
    if not os.path.exists(p):
        clips.setdefault(uid, {})["H_error"] = "no vimo output"
        continue
    v = np.load(p)
    top, bottom = v["top0"], v["bottom0"]
    TC.sync(); t0 = time.perf_counter()
    out = v1action.predict(hm, top, bottom, src_fps=TC.FPS)
    TC.sync(); clips[uid]["H_v1_hits"] = time.perf_counter() - t0
    clips[uid]["n_native_frames"] = int(len(v["frames"]))
    clips[uid]["slots_cov"] = {s: float(np.isfinite(v[s][:, 0, 0]).mean()) for s in ("top0", "top1", "bottom0", "bottom1")}

res = {"phase": "D", "hw": info, "load_s": load, "warmup_uid": warm,
       "warmup": {warm: clips.get(warm)}, "clips": {u: clips.get(u) for u in test}}
TC.dump("phase_d", res)
