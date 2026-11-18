"""Thin two-player wrapper around GVHMR's tools/demo/demo.py (static camera).

Per clip:
  1. YOLOv8x tracking exactly as GVHMR's Tracker.track (conf 0.5, class person).
  2. Track selection: GVHMR ranks tracks by summed normalised box area.  We keep that
     ranking but restrict it to tracks whose foot point (box bottom-centre, back-projected
     to the ground with our court calibration) is inside the court (+margin) for a minimum
     number of frames, and take up to 2 players per half (singles: 1, doubles: 2).
     Temporally disjoint, spatially continuous same-half fragments are appended to a
     player (ID-switch repair); co-located duplicate tracks are rejected.
  3. For each selected player: GVHMR's own box processing (interpolate gaps, 2x moving
     average, get_bbx_xys_from_xyxy(1.2)), ViTPose, HMR2 features, then DemoPL.predict
     with static_cam=True and K from our calibration.  The network only sees the frames
     of the player's span [first, last] (no edge-padded boxes).
  4. Camera-frame SMPL-24 joints from smpl_params_incam via GVHMR's body model
     (SMPL-X supermotion -> smplx2smpl_sparse -> smpl_neutral_J_regressor), as in
     demo.render_incam / render_global.
"""
import argparse, json, os, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2
import torch

GV = Path("/scratch/ciro/gvhmr/GVHMR")
os.chdir(GV)
sys.path.insert(0, str(GV))

import hydra
from hydra import initialize_config_module, compose
from hmr4d.configs import register_store_gvhmr
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import get_video_lwh, read_video_np
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor
from hmr4d.utils.preproc.vitfeat_extractor import crop_and_resize, IMAGE_MEAN, IMAGE_STD
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.utils.seq_utils import get_frame_id_list_from_mask, linear_interpolate_frame_ids, frame_id_to_mask, rearrange_by_mask
from hmr4d.utils.net_utils import moving_average_smooth, detach_to_cpu, to_cuda
from hmr4d.utils.smplx_utils import make_smplx
from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL  # registers model/gvhmr/gvhmr_pl_demo in the config store (as demo.py does)

COURT_W, COURT_L, NET_Y = 6.1, 13.4, 6.7
MARGIN_X, MARGIN_Y = 0.75, 2.0  # players lunge past side lines / retreat behind baselines
MAX_PER_SIDE = 2  # CCTV test set contains doubles (d1026_*) as well as singles


def load_K(calib_path, W, H):
    cam = json.load(open(calib_path))["cam"]
    sw, sh = cam["size"]
    s = W / sw
    assert abs(H / sh - s) < 1e-3, (W, H, cam["size"])
    f = cam["f"] * s
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], dtype=np.float64)
    R, _ = cv2.Rodrigues(np.asarray(cam["rvec"], dtype=np.float64))
    t = np.asarray(cam["tvec"], dtype=np.float64)
    return K, R, t, cam


def ground_from_pixel(uv, K, R, t):
    """uv (N,2) -> court ground points (N,2) on z=0 (nan if ray goes up)."""
    uv1 = np.concatenate([uv, np.ones((len(uv), 1))], 1)
    d = (R.T @ (np.linalg.inv(K) @ uv1.T)).T
    C = -R.T @ t
    lam = -C[2] / d[:, 2]
    P = C[None] + lam[:, None] * d
    P[lam <= 0] = np.nan
    return P[:, :2]


def in_court(xy):
    x, y = xy[:, 0], xy[:, 1]
    return (x > -MARGIN_X) & (x < COURT_W + MARGIN_X) & (y > -MARGIN_Y) & (y < COURT_L + MARGIN_Y)


def select_players(track_history, L, W, H, K, R, t, min_frames):
    ids_f, ids_b = defaultdict(list), defaultdict(list)
    for fi, fr in enumerate(track_history):
        for d in fr:
            ids_f[d["id"]].append(fi)
            ids_b[d["id"]].append(d["bbx_xyxy"])
    info = {}
    for k in ids_f:
        f = np.array(ids_f[k]); b = np.array(ids_b[k], dtype=np.float64)
        foot = np.stack([(b[:, 0] + b[:, 2]) / 2, b[:, 3]], 1)
        g = ground_from_pixel(foot, K, R, t)
        ok = in_court(g) & np.isfinite(g).all(1)
        area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) / W / H
        info[k] = dict(frames=f, boxes=b, ground=g, inc=ok,
                       score=float(area[ok].sum()), n_in=int(ok.sum()),
                       side=("near" if np.nanmedian(g[ok, 1]) < NET_Y else "far") if ok.any() else None)
    cands = [k for k in info if info[k]["n_in"] >= min_frames]
    players = []
    for side in ["near", "far"]:  # up to MAX_PER_SIDE players per half (2 = doubles)
        sc = sorted([k for k in cands if info[k]["side"] == side], key=lambda k: -info[k]["score"])
        slots = []  # each: dict(members, g={frame: ground xy})
        for k in sc:
            I = info[k]
            gk = {int(f): g for f, g, ok in zip(I["frames"], I["ground"], I["inc"]) if ok}
            fk = np.array(sorted(gk))
            placed = False
            for sl in slots:  # ID-switch repair: temporally disjoint and spatially continuous
                if len(set(gk) & set(sl["g"])) > 2:
                    continue
                uf = np.array(sorted(sl["g"]))
                ok = False
                for fe in (fk[0], fk[-1]):  # compare fragment ends with the slot's nearest-in-time frame
                    a = int(uf[np.abs(uf - fe).argmin()])
                    if abs(a - fe) <= 30 and np.linalg.norm(sl["g"][a] - gk[int(fe)]) < 1.5:
                        ok = True
                if ok:
                    sl["members"].append(k); sl["g"].update({f: g for f, g in gk.items() if f not in sl["g"]})
                    placed = True; break
            if not placed and len(slots) < MAX_PER_SIDE:
                # reject duplicate detections of an already-selected player (co-located while co-visible)
                dup = False
                for sl in slots:
                    com = sorted(set(gk) & set(sl["g"]))
                    if com and np.median([np.linalg.norm(gk[f] - sl["g"][f]) for f in com]) < 0.8:
                        dup = True
                if not dup:
                    slots.append(dict(members=[k], g=gk))
        for sl in slots:
            players.append(dict(side=side, members=sl["members"]))
    # gather per-player detected boxes (in-court frames only, first member wins on overlap)
    out = []
    for p in players:
        fb = {}
        for k in p["members"]:
            I = info[k]
            for fi, b, ok in zip(I["frames"], I["boxes"], I["inc"]):
                if ok and fi not in fb:
                    fb[int(fi)] = b
        fr = np.array(sorted(fb)); bx = np.stack([fb[i] for i in fr])
        out.append(dict(side=p["side"], track_ids=p["members"], det_frames=fr, det_boxes=bx))
    summary = {int(k): dict(n=len(v["frames"]), n_in=v["n_in"], score=round(v["score"], 3), side=v["side"]) for k, v in info.items()}
    return out, summary


def gvhmr_box_track(det_frames, det_boxes):
    """GVHMR Tracker.get_one_track post-processing, restricted to span [first, last]."""
    s, e = int(det_frames[0]), int(det_frames[-1]) + 1
    n = e - s
    mask = frame_id_to_mask(torch.tensor(det_frames - s), n)
    bb = rearrange_by_mask(torch.tensor(det_boxes, dtype=torch.float32), mask)
    bb = linear_interpolate_frame_ids(bb, get_frame_id_list_from_mask(~mask))
    assert (bb.sum(1) != 0).all()
    bb = moving_average_smooth(bb, window_size=5, dim=0)
    bb = moving_average_smooth(bb, window_size=5, dim=0)
    return s, e, bb, mask.numpy()


def get_batch_frames(imgs_ds, bbx_xys, img_ds=0.5, img_dst_size=256):
    """Same as hmr4d vitfeat_extractor.get_batch(path_type='video') but on pre-read frames."""
    gt_center = bbx_xys[:, :2]; gt_bbx_size = bbx_xys[:, 2]
    ds_factors = ((gt_bbx_size * img_ds) / img_dst_size / 2.0).numpy()
    imgs = np.stack([cv2.GaussianBlur(v, (5, 5), (d - 1) / 2) if d > 1.1 else v for v, d in zip(imgs_ds, ds_factors)])
    il, bl = [], []
    for i in range(len(imgs)):
        img, b = crop_and_resize(imgs[i], gt_center[i] * img_ds, gt_bbx_size[i] * img_ds, img_dst_size, enlarge_ratio=1.0)
        il.append(img); bl.append(b)
    imgs = torch.from_numpy(np.stack(il))
    bbx = torch.from_numpy(np.stack(bl)) / img_ds
    imgs = ((imgs / 255.0 - IMAGE_MEAN) / IMAGE_STD).permute(0, 3, 1, 2)
    return imgs, bbx

SMPLX_NPZ = GV / "inputs/checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz"


def preprocess_clip(uid, a, tracker, vitpose_ex, feat_ex, gpu):
    """Tracking + selection + ViTPose + HMR2 features (no SMPL-X needed). Cached to disk."""
    vp = f"{a.clips}/{uid}.mp4"
    torch.cuda.synchronize(); tc = time.time(); tm = {}
    L, W, H = get_video_lwh(vp)
    fps = cv2.VideoCapture(vp).get(cv2.CAP_PROP_FPS)
    K, R, t, cam = load_K(Path(a.calib) / f"{uid}.json", W, H)
    with torch.no_grad():
        ts = time.time(); th = tracker.track(vp); tm["track"] = time.time() - ts
    ts = time.time(); imgs_ds = read_video_np(vp, scale=0.5); tm["read"] = time.time() - ts
    if len(imgs_ds) != L:  # container frame count can exceed the decodable frames (DroneData back videos)
        assert len(imgs_ds) < L and len(th) >= len(imgs_ds), (len(imgs_ds), L, len(th))
        L = len(imgs_ds); th = th[:L]
    players, tsum = select_players(th, L, W, H, K, R, t, max(10, int(a.min_frames_frac * L)))
    P = []
    for pi, p in enumerate(players):
        s, e, bb, detm = gvhmr_box_track(p["det_frames"], p["det_boxes"])
        bxys = get_bbx_xys_from_xyxy(bb, base_enlarge=1.2).float()
        with torch.no_grad():
            ts = time.time()
            imgs, bx_c = get_batch_frames(imgs_ds[s:e], bxys)
            kp2d = vitpose_ex.extract(imgs, bx_c)
            feats = feat_ex.extract_video_features(imgs, bx_c)
            torch.cuda.synchronize(); tm[f"p{pi}_prep"] = time.time() - ts
        P.append(dict(side=p["side"], track_ids=p["track_ids"], s=s, e=e, bbx_xyxy=bb, detected=detm,
                      bbx_xys=bxys, kp2d=kp2d.cpu(), feats=feats.cpu()))
    torch.cuda.synchronize(); tm["wall_prep"] = time.time() - tc
    return dict(uid=uid, L=L, W=W, H=H, fps=fps, K=K, R=R, t=t, players=P, track_summary=tsum, track_history=th, timing=tm, gpu=gpu)


def hmr4d_clip(pp, model, smplx, smplx2smpl, J_reg):
    torch.cuda.synchronize(); tc = time.time()
    K, R, t = pp["K"], pp["R"], pp["t"]
    res = dict(uid=pp["uid"], n_frames=pp["L"], fps=pp["fps"], W=pp["W"], H=pp["H"], K=K,
               K_source="calib_bc (f scaled to video size, principal point at image centre)",
               cam_R_court2cam=R, cam_t_court2cam=t, gpu=pp["gpu"], n_players=len(pp["players"]),
               track_summary=json.dumps(pp["track_summary"]),
               joint_set="SMPL-24 (smplx->smpl verts via smplx2smpl_sparse, smpl_neutral_J_regressor)",
               units="metres; joints_cam in OpenCV camera frame (x right, y down, z forward)")
    tm = {}
    for pi, p in enumerate(pp["players"]):
        n = p["e"] - p["s"]
        data = {"length": torch.tensor(n), "bbx_xys": p["bbx_xys"], "kp2d": p["kp2d"],
                "K_fullimg": torch.from_numpy(K).float()[None].repeat(n, 1, 1),
                "cam_angvel": compute_cam_angvel(torch.eye(3).repeat(n, 1, 1)),
                "f_imgseq": p["feats"]}
        with torch.no_grad():
            ts = time.time()
            pred = detach_to_cpu(model.predict(data, static_cam=True))
            Js = {}
            for kk in ["incam", "global"]:
                out = smplx(**to_cuda(pred[f"smpl_params_{kk}"]))
                v = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in out.vertices])
                Js[kk] = torch.einsum("jv,lvc->ljc", J_reg, v).cpu().numpy()
            torch.cuda.synchronize(); tm[f"p{pi}_hmr4d"] = time.time() - ts
        J = Js["incam"]
        Jw = (R.T @ (J.reshape(-1, 3).T - t[:, None])).T.reshape(J.shape)  # court frame via our calib
        pr = f"p{pi}_"
        res.update({pr + "side": p["side"], pr + "track_ids": np.array(p["track_ids"]),
                    pr + "frames": np.arange(p["s"], p["e"]), pr + "detected": p["detected"],
                    pr + "bbx_xyxy": p["bbx_xyxy"].numpy(), pr + "bbx_xys": p["bbx_xys"].numpy(),
                    pr + "kp2d_vitpose": p["kp2d"].numpy(),
                    pr + "joints_cam": J.astype(np.float32), pr + "joints_court": Jw.astype(np.float32),
                    pr + "joints_global_gvhmr": Js["global"].astype(np.float32)})
        for kk in ["incam", "global"]:
            for pk, pv in pred[f"smpl_params_{kk}"].items():
                res[f"{pr}smplx_{kk}_{pk}"] = pv.numpy()
    torch.cuda.synchronize(); tm["wall_hmr4d"] = time.time() - tc
    tm.update(pp["timing"])
    tm["wall_clip"] = tm["wall_prep"] + tm["wall_hmr4d"]
    tm["video_sec"] = pp["L"] / pp["fps"]
    tm["sec_per_video_min"] = tm["wall_clip"] / (tm["video_sec"] / 60)
    res["timing"] = json.dumps(tm)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", nargs="+", required=True)
    ap.add_argument("--clips", default="/scratch/ciro/birdseye_v2/repo/data/cctv/clips")
    ap.add_argument("--calib", default="/scratch/ciro/gvhmr/calib")
    ap.add_argument("--prep", default="/scratch/ciro/gvhmr/prep_cctv")
    ap.add_argument("--out", default="/scratch/ciro/gvhmr/out_cctv")
    ap.add_argument("--min_frames_frac", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True); Path(a.prep).mkdir(parents=True, exist_ok=True)
    have_smplx = SMPLX_NPZ.exists()
    if not have_smplx:
        Log.warn(f"{SMPLX_NPZ} missing: running preprocessing only (no HMR4D / joints)")
    gpu = torch.cuda.get_device_name()
    tracker = vitpose_ex = feat_ex = model = None

    for uid in a.uids:
        op = Path(a.out) / f"{uid}.npz"; pp_path = Path(a.prep) / f"{uid}.pt"
        if op.exists() and not a.overwrite:
            Log.info(f"skip {uid}"); continue
        if not Path(f"{a.clips}/{uid}.mp4").exists() or not (Path(a.calib) / f"{uid}.json").exists():
            Log.warn(f"{uid}: missing video or calib"); continue
        if pp_path.exists() and not a.overwrite:
            pp = torch.load(pp_path)
        else:
            if tracker is None:
                t0 = time.time()
                tracker = Tracker(); vitpose_ex = VitPoseExtractor(tqdm_leave=False); feat_ex = Extractor(tqdm_leave=False)
                Log.info(f"preproc models loaded in {time.time()-t0:.1f}s on {gpu}")
            pp = preprocess_clip(uid, a, tracker, vitpose_ex, feat_ex, gpu)
            torch.save(pp, pp_path)
        Log.info(f"{uid}: prep L={pp['L']} players={len(pp['players'])} sides={[p['side'] for p in pp['players']]} "
                 f"spans={[(p['s'], p['e'], int(p['detected'].sum())) for p in pp['players']]} timing={pp['timing']}")
        if not have_smplx:
            continue
        if model is None:
            with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
                register_store_gvhmr()
                cfg = compose(config_name="demo", overrides=["video_name=x", "static_cam=True"])
            model = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().cuda()
            smplx = make_smplx("supermotion").cuda()
            smplx2smpl = torch.load("hmr4d/utils/body_model/smplx2smpl_sparse.pt").cuda()
            J_reg = torch.load("hmr4d/utils/body_model/smpl_neutral_J_regressor.pt").cuda()
        res = hmr4d_clip(pp, model, smplx, smplx2smpl, J_reg)
        np.savez_compressed(op, **res)
        depths = [float(np.median(res[f"p{i}_joints_cam"][:, 0, 2])) for i in range(res["n_players"])]
        Log.info(f"{uid}: players={res['n_players']} pelvis_depth_med={depths} timing={res['timing']}")


if __name__ == "__main__":
    main()
