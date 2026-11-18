"""Phase C (bev2_env): CPU fits + our student network.
  C2f  BirdsCourt camera fit (tools/fb_birdscourt.py fit)
  C1r  Birdseye v1 RANSAC self-calibration (1:Court/court_detection.py fit_self_calibrated, n_iters=100 as
       calibrate_from_own_detections.py) on the MonoTrack 24 points (poles dropped), VGGT-1B hfov as focal hint
  O    rally observation arrays (tools/build_obs.py run, SET=cctv)
  N    student network: tools/train_rallynet.py load_feats + infer (ARCH=rallyphys STUDENT=1 ... TAG=t3)
  SP   shot segmentation shared by the v1 / MonoTrack shuttle baselines (tools/xview_eval.py baselines_one)
  V1f  Birdseye v1 per-shot physics fit (tools/eval_shuttle.py fit_v1), evaluated at every frame
  MTf  MonoTrack per-shot fit (birdseye_v2.monotrack_fit.reconstruct), evaluated at every frame"""
import json, os, sys, time

os.environ.update(FB_SET="cctv", ARCH="rallyphys", STUDENT="1", BODY24="20", B24_SHAPE="5", FREE_S="1", TAG="t3",
                  OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"))
sys.path.insert(0, "/scratch/ciro/timing")
import tcommon as TC  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, TC.ROOT)
sys.path.insert(0, os.path.join(TC.ROOT, "tools"))
sys.path.insert(0, os.path.join(TC.ROOT, "third_party"))
sys.argv = ["build_obs.py", "cctv"]
import fb_birdscourt as BC  # noqa: E402
import build_obs as BO  # noqa: E402
import train_rallynet as TR  # noqa: E402
from birdseye_v2 import shuttle3d as S3  # noqa: E402
import eval_shuttle as ES  # noqa: E402
from birdseye_v2.court import Camera  # noqa: E402
from birdseye_v2.players import foot_point, load_pose  # noqa: E402
from birdseye_v2.monotrack_fit import reconstruct as mt_reconstruct  # noqa: E402
import cctv_tri as CT  # noqa: E402

S3.FPS = TC.FPS
ES.FPS = TC.FPS
sys.path.insert(0, "/project/6101776/ciro/Badminton LLM/Birdseye/1:Court")
from court_detection import court_detection as V1Court  # noqa: E402

OUT = os.path.join(TC.ROOT, "outputs", "cctv")
NAMES = ["P1_TL", "P2_BL", "P3_BR", "P4_TR", "poleL_top", "poleR_top", "P15_netL", "P16_netR",
         "P5_singlesTL", "P6_singlesBL", "P7_singlesBR", "P8_singlesTR",
         "P9_serviceFarL", "P10_serviceFarR", "P11_serviceNearL", "P12_serviceNearR",
         "P13_serviceFarCenter", "P14_serviceNearCenter",
         "P17_doublesFarL", "P18_doublesFarR", "P19_doublesNearL", "P20_doublesNearR",
         "P21_baseFarCenter", "P22_baseNearCenter"]  # 1:Court/calibrate_from_own_detections.py


def baseline_prep(uid, cam):
    """tools/xview_eval.py baselines_one, the part shared by v1 and MonoTrack (minus the side-camera scoring)."""
    tr = CT.track(os.path.join(OUT, "refined", uid + ".json"))
    if len(tr) < 10:
        return None
    frames = np.array(sorted(tr), float)
    px = np.array([tr[k] for k in sorted(tr)], float)
    poses = load_pose(os.path.join(OUT, "pose1280", uid + ".npz"))
    lf = S3.landing_frame(frames, px)
    f_c, p_c = S3.clean_track(frames, px, [lf - 1] if lf is not None else [frames[-1]])
    hits = list(S3.detect_hits(f_c, p_c, cam=cam, poses=poses))
    if lf is not None:
        hits = [h for h in hits if h < lf - 2]
    if not hits or hits[0] > f_c[0] + 4:
        hits = [float(f_c[0])] + hits
    frames, px = S3.clean_track(frames, px, hits)
    shots = S3.segment_shots(frames, hits, px)
    shots = S3.merge_explained(cam, frames, px, S3.split_unexplained(cam, frames, px, shots, max_rms=16.0), max_rms=16.0)
    hands = [ES.hands_for(cam, poses, frames, px, a) for a, _ in shots]
    return frames, px, poses, shots, hands


def make_foot_at(cam, frames, px, poses):
    def foot_at(f):
        f = int(round(f))
        uv = ES.interp_px(frames, px, f)
        if not (0 <= f < len(poses)) or uv is None:
            return None
        best, bd = None, 1e9
        for kp in poses[f]:
            k = kp[kp[:, 2] > 0.3]
            if len(k) >= 3 and np.linalg.norm(k[:, :2].mean(0) - uv) < bd:
                bd, best = np.linalg.norm(k[:, :2].mean(0) - uv), kp
        return None if best is None else foot_point(cam, best)
    return foot_at


info = TC.hw_info()
load = {}
t0 = time.perf_counter(); model = TR.load_model(); TC.sync(); load["N_student_t3"] = time.perf_counter() - t0
nparam = sum(p.numel() for p in model.parameters())
warm, test = TC.uids()
res = {"phase": "C", "hw": info, "load_s": load, "student_params": int(nparam), "clips": {}, "warmup": {}}
for k, uid in enumerate([warm] + test):
    src = TC.clip(uid)
    TC.predecode([src])
    n = TC.n_frames(src)
    T = TC.Timer()
    extra = {}
    # C2 fit
    with T("C2_birdscourt_fit"):
        _, st = BC.fit(uid)
    extra["C2_valid"] = bool(st and st.get("valid"))
    # C1 v1 RANSAC with VGGT focal hint
    mp = os.path.join(OUT, "mt_court", uid + ".pts")
    vj = os.path.join(OUT, "vggt_" + uid + ".json")
    if os.path.exists(mp):
        lines = [l.strip() for l in open(mp) if l.strip()]
        pts = [[float(v) for v in l.split(";")] for l in lines]
        hfov = json.load(open(vj))["hfov_deg"] if os.path.exists(vj) else None
        extra["C1_hfov"] = hfov
        if len(pts) == len(NAMES):
            non_pole = {a: b for a, b in zip(NAMES, pts) if "pole" not in a and np.all(np.isfinite(b))}
            w, h = TC.CACHE[os.path.abspath(src)][2]
            try:
                with T("C1_v1_ransac_fit"):
                    cd = V1Court(width=w, height=h)
                    cd.set_detections(non_pole)
                    rms, inl, outl = cd.fit_self_calibrated(inlier_thresh_px=20.0, n_iters=100, focal_hint_hfov_deg=hfov)
                extra["C1_rms_px"] = float(rms); extra["C1_inliers"] = len(inl)
            except Exception as e:
                extra["C1_error"] = repr(e)[:200]
        else:
            extra["C1_error"] = f"{len(pts)} points"
    else:
        extra["C1_error"] = "no MonoTrack output"
    # O
    dst = os.path.join(TC.ROOT, "outputs", "rally_obs", "cctv", uid + ".npz")
    if os.path.exists(dst):
        os.remove(dst)
    with T("O_build_obs"):
        msg = BO.run(uid)
    extra["O_msg"] = msg
    # N
    if os.path.exists(dst):
        with T("N_load_feats"):
            f = TR.load_feats(dst)
        with T("N_forward"):
            with torch.no_grad():
                out = TR.infer(model, f)
    # baselines' shuttle fits
    cam = Camera.from_dict(json.load(open(os.path.join(OUT, "calib_bc", uid + ".json")))["cam"])
    with T("SP_shot_prep"):
        bp = baseline_prep(uid, cam)
    if bp is not None:
        frames, px, poses, shots, hands = bp
        tq = np.arange(n, dtype=float)
        extra["n_shots"] = len(shots)
        with T("V1_shuttle_fit"):
            S1 = ES.fit_v1(cam, frames, px, shots, hands)(tq)
        foot_at = make_foot_at(cam, frames, px, poses)
        with T("MT_shuttle_fit"):
            S2 = mt_reconstruct(cam, frames, px, [int(round(a)) for a, _ in shots] + [int(round(shots[-1][1]))], foot_at, TC.FPS)(tq)
        extra["V1_cov"] = float(np.isfinite(S1[:, 0]).mean()); extra["MT_cov"] = float(np.isfinite(S2[:, 0]).mean())
    TC.drop(src)
    rec = {"n_frames": n, "video_s": n / TC.FPS, **{k_: round(v, 4) for k_, v in T.t.items()}, **extra}
    (res["warmup"] if k == 0 else res["clips"])[uid] = rec
    print(uid, json.dumps(rec), flush=True)
    TC.dump("phase_c", res)
