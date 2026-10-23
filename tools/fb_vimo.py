"""VIMO (TRAM) 3D bodies for the two players of each FBBench clip, as in Birdseye v1
(2:Player/run_vimo_bfmd_selfdetect.py): per-side person boxes from our tracked YOLOv8x-pose
detections, VIMO with our calibrated camera, joints to world, then v1's ground correction
(ray-cast a pixel-stable ankle to the floor -- at ankle height, see players.ANKLE_H -- and rigidly
shift the body). Native 25 fps frames (FBBench's repeated frames are dropped).

Needs third_party/tram/data/smpl/SMPL_NEUTRAL.pkl (licensed) and data/pretrain/vimo_checkpoint.pth.tar.
Run in .venv_vimo (timm 0.6.7).
Output: outputs/fb/vimo/<uid>.npz  {top, bottom: (T,24,3) world joints (NaN = missing), frames: native idx}
usage: .venv_vimo/Scripts/python tools/fb_vimo.py [uid ...]"""
import glob, os, shutil, sys, tempfile, time, types

import cv2
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True  # fp32 matmul with TF32 off stalls on the RTX 5080
torch.backends.cudnn.allow_tf32 = False
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TRAM = os.path.join(ROOT, "third_party", "tram")
sys.path.insert(0, ROOT)
sys.path.insert(0, TRAM)

# lib.pipeline.tools pulls in detectron2 / SAM / DEVA for TRAM's own detection; VIMO inference only
# needs parse_chunks from it. Provide just that.
_pt = types.ModuleType("lib.pipeline.tools")


def parse_chunks(frame, boxes, min_len=16):
    step = np.concatenate([[0], frame[1:] - frame[:-1]])
    breaks = list(np.where(step != 1)[0][1:]) + [len(frame)]
    fc, bc, start = [], [], 0
    for bk in breaks:
        if bk - start >= min_len:
            fc.append(frame[start:bk]); bc.append(boxes[start:bk])
        start = bk
    return fc, bc


_pt.parse_chunks = parse_chunks
sys.modules["lib.pipeline"] = types.ModuleType("lib.pipeline")
sys.modules["lib.pipeline.tools"] = _pt

from birdseye_v2 import camtrack
from birdseye_v2.broadcast import native_frames
from birdseye_v2.court import NET_Y, Camera
from birdseye_v2.players import ANKLE_H, load_pose, players_3d

FB_SET = os.environ.get("FB_SET", "bench")
CLIPS = os.path.join(ROOT, "data", *{"bench": ("fbbench", "hf_video_clips_qa"), "train": ("fb20k", "clips"),
                                     "bfmd": ("bfmd", "clips")}[FB_SET])  # bfmd: BFMD singles rally clips
OUT = os.path.join(ROOT, "outputs", {"bench": "fb", "train": "fb20k", "bfmd": "bfmd"}[FB_SET])
CALIB_DIR = os.environ.get("CALIB_DIR", "calib_bc")  # BirdsCourt (tools/fb_birdscourt.py); "calib" = old fb_calibrate
VIMO_DIR = os.environ.get("VIMO_DIR", "vimo_bc")
CAMTRACK = os.environ.get("CAMTRACK", "1") == "1"  # use <out>/camtrack/<uid>.json per-frame cameras when present
POSE_DIR = os.environ.get("POSE_DIR", "pose")  # "pose1280": YOLOv8x-pose at imgsz 1280 (768 missed ~1/4 of far-player frames)
STABLE_PIX_1080 = 6.0  # v1's threshold at 1080p; scaled with image height
MAD_K, MAX_FOOT_H = 5.0, 1.1


def fast_chunk(model, imgfiles, boxes, img_focal, img_center, bs=32, wb=8):
    """Same result as HMR_VIMO.inference_chunk, but the ViT-H backbone runs ONCE per frame and its
    features are reused by every 16-frame window (the original recomputes all 16 crops per output frame).
    Output selection per window is copied from inference_chunk."""
    import einops
    from lib.datasets.track_dataset import TrackDataset
    from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat
    from torch.utils.data import default_collate
    dev = "cuda"
    db = TrackDataset(imgfiles, boxes, img_focal=img_focal, img_center=img_center, normalization=True, dilate=1.2)
    batch = default_collate([db[i] for i in range(len(db))])
    N = len(db)
    center, scale = batch["center"].to(dev), batch["scale"].to(dev)
    focal, icenter = batch["img_focal"].to(dev), batch["img_center"].to(dev)
    feats = []
    with torch.no_grad():
        for a in range(0, N, bs):
            with torch.autocast("cuda", dtype=torch.float16):
                feats.append(model.backbone(batch["img"][a:a + bs].to(dev)[:, :, :, 32:-32]).float())
        feats = torch.cat(feats)                                   # (N,1280,16,12)
        bbox_info = model.bbox_est(center, scale, focal, icenter)  # (N,3)
        starts = [0] if N == 16 else list(range(0, N - 15))
        keep_idx = []  # (window, frame-in-window) pairs to output, as inference_chunk
        for w, s0 in enumerate(starts):
            i = s0 + 15
            if N == 16:
                keep_idx += [(w, k) for k in range(16)]
            elif i == 15:
                keep_idx += [(w, k) for k in range(9)]
            elif i == N - 1:
                keep_idx += [(w, k) for k in range(8, 16)]
            else:
                keep_idx.append((w, 8))
        outs = {k: [] for k in ("pred_shape", "pred_rotmat", "trans_full")}
        for b0 in range(0, len(starts), wb):
            ws = starts[b0:b0 + wb]
            idx = torch.tensor([s0 + t for s0 in ws for t in range(16)], device=dev)
            f = feats[idx]
            bb = bbox_info[idx]
            if model.st_module is not None:
                f = torch.cat([f, einops.repeat(bb, "b c -> b c h w", h=16, w=12)], 1)
                f = einops.rearrange(f, "(b t) c h w -> (b h w) t c", t=16)
                f = model.st_module(f)
                f = einops.rearrange(f, "(b h w) t c -> (b t) c h w", h=16, w=12)
            pose, shape, cam = model.smpl_head(f)
            if model.motion_module is not None:
                p = torch.cat([einops.rearrange(pose, "(b t) c -> b t c", t=16), einops.rearrange(bb, "(b t) c -> b t c", t=16)], 2)
                pose = einops.rearrange(model.motion_module(p), "b t c -> (b t) c")
            rotmat = rot6d_to_rotmat(pose).reshape(-1, 24, 3, 3)
            trans = model.get_trans(cam, center[idx], scale[idx], focal[idx], icenter[idx])
            for w_local in range(len(ws)):
                for (w, k) in [x for x in keep_idx if x[0] == b0 + w_local]:
                    j = w_local * 16 + k
                    outs["pred_shape"].append(shape[j:j + 1].cpu()); outs["pred_rotmat"].append(rotmat[j:j + 1].cpu())
                    outs["trans_full"].append(trans[j:j + 1].cpu())
    return {k: torch.cat(v) for k, v in outs.items()}


def per_camera_inference(model, files, boxes, valid, cams, step_rel=0.03, span_rel=0.2, move_m=1.5):
    """fast_inference per run of frames from one camera: a run ends where the focal length jumps more than step_rel
    from the previous frame (a cut), where the run's focal range passes span_rel (a long zoom; low courtside
    cameras zoom smoothly as they follow play -- comparing with the run's first frame at 5% cut those into runs
    shorter than VIMO's 16 frames), or where the camera centre moves more than move_m (another court camera --
    side, corner; a pan keeps its centre)."""
    T = len(files)
    seg, start, lo, hi = [], None, None, None
    for i in range(T + 1):
        c = cams[i] if i < T else None
        if start is not None:
            p = cams[i - 1]
            if (c is None or abs(c.f - p.f) > step_rel * p.f or max(hi, c.f) > (1 + span_rel) * min(lo, c.f)
                    or np.linalg.norm(c.center - cams[start].center) > move_m):
                seg.append((start, i)); start = None
            else:
                lo, hi = min(lo, c.f), max(hi, c.f)
        if c is not None and start is None:
            start, lo, hi = i, c.f, c.f
    out = {"pred_shape": [], "pred_rotmat": [], "pred_trans": [], "frame": []}
    for a, b in seg:
        v = valid.copy(); v[:a] = False; v[b:] = False
        if v.sum() < 16:
            continue
        f = float(np.median([cams[i].f for i in range(a, b)]))
        r = fast_inference(model, files, boxes, v, f, np.array(cams[a].size, np.float32) / 2)
        if r is not None:
            for k in out:
                out[k].append(r[k])
    return {k: torch.cat(v) for k, v in out.items()} if out["frame"] else None


MAX_GAP = 8  # frames: a player seen only every other frame still gets a body (VIMO needs 16+ consecutive boxes)


def fill_box_gaps(boxes, valid, max_gap=MAX_GAP):
    """Linearly interpolate the box over gaps of <= max_gap frames between two valid frames."""
    f = np.where(valid)[0]
    for a, b in zip(f[:-1], f[1:]):
        if 1 < b - a <= max_gap + 1:
            w = (np.arange(a + 1, b) - a)[:, None] / (b - a)
            boxes[a + 1:b] = (1 - w) * boxes[a] + w * boxes[b]
            valid[a + 1:b] = True
    return boxes, valid


def fast_inference(model, imgfiles, boxes, valid, img_focal, img_center):
    frame = np.arange(len(imgfiles))[valid]
    fc, bc = parse_chunks(frame, boxes[valid], min_len=16)
    if not fc:
        return None
    res = {"pred_shape": [], "pred_rotmat": [], "pred_trans": [], "frame": []}
    for f, b in zip(fc, bc):
        r = fast_chunk(model, np.array(imgfiles)[f], b, img_focal, img_center)
        res["pred_shape"].append(r["pred_shape"]); res["pred_rotmat"].append(r["pred_rotmat"])
        res["pred_trans"].append(r["trans_full"]); res["frame"].append(torch.from_numpy(f))
    return {k: torch.cat(v) for k, v in res.items()}


def side_boxes_image(poses, region):
    """Calibration-free: the people standing inside the court region (green surface, widened); the
    bottom player = lowest feet in the image, the top player = highest."""
    T = len(poses)
    out = {s: {"box": np.zeros((T, 4), np.float32), "valid": np.zeros(T, bool), "ankle": np.full((T, 2), np.nan)}
           for s in ("top", "bottom")}
    h, w = region.shape
    for i, kps in enumerate(poses):
        ppl = []
        for kp in kps:
            k = kp[kp[:, 2] > 0.3]
            if len(k) < 5:
                continue
            a = kp[[15, 16]]
            a = a[a[:, 2] > 0.3]
            foot = a[:, :2].mean(0) if len(a) else k[np.argmax(k[:, 1]), :2]
            fx, fy = int(np.clip(foot[0], 0, w - 1)), int(np.clip(foot[1], 0, h - 1))
            if region[fy, fx]:
                x1, y1 = k[:, :2].min(0); x2, y2 = k[:, :2].max(0)
                ppl.append((foot, (x1, y1, x2, y2), a[:, :2].mean(0) if len(a) else np.full(2, np.nan), (y2 - y1)))
        if not ppl:
            continue
        ppl.sort(key=lambda p: p[0][1])
        for s, p in (("top", ppl[0]), ("bottom", ppl[-1])) if len(ppl) > 1 else ((("bottom" if ppl[0][0][1] > h * 0.6 else "top"), ppl[0]),):
            x1, y1, x2, y2 = p[1]
            pad = 0.1 * max(x2 - x1, y2 - y1)
            out[s]["box"][i] = [x1 - pad, y1 - pad, x2 + pad, y2 + pad]
            out[s]["valid"][i] = True
            out[s]["ankle"][i] = p[2]
    return out


def nominal_camera(size=(1280, 720)):
    """Typical BWF broadcast camera (from a well-calibrated FBBench clip, video_000651): ~13.6 m behind
    the near baseline, 5.5 m up, f ~ 2000 px. Used when a clip's own calibration is not trusted; only
    the rotation matters for v1's pelvis-relative features."""
    from birdseye_v2.court import init_from_corners
    return init_from_corners([(219, 645), (1062, 648), (866, 278), (414, 278)], size, 2000.0)


def side_boxes(cam, poses):
    """Per native frame and side: box (x1,y1,x2,y2), valid flag, mid-ankle px. One person per side:
    the on-court person in that half nearest to the previous pick. cam: one Camera, or one per frame
    (None = no court view in that frame: no box)."""
    cams = cam if isinstance(cam, list) else [cam] * len(poses)
    T = len(poses)
    out = {s: {"box": np.zeros((T, 4), np.float32), "valid": np.zeros(T, bool), "ankle": np.full((T, 2), np.nan)}
           for s in ("top", "bottom")}
    last = {"top": None, "bottom": None}
    for i, kps in enumerate(poses):
        if cams[i] is None:
            continue
        ppl = players_3d(cams[i], kps)
        for s in ("top", "bottom"):
            cand = [p for p in ppl if (p["foot"][1] < NET_Y) == (s == "bottom")]
            if not cand:
                continue
            p = min(cand, key=lambda q: 0 if last[s] is None else np.linalg.norm(q["foot"] - last[s]))
            last[s] = p["foot"]
            k = p["kp"][p["kp"][:, 2] > 0.3]
            x1, y1 = k[:, :2].min(0); x2, y2 = k[:, :2].max(0)
            pad = 0.1 * max(x2 - x1, y2 - y1)
            out[s]["box"][i] = [x1 - pad, y1 - pad, x2 + pad, y2 + pad]
            out[s]["valid"][i] = True
            a = p["kp"][[15, 16]]
            a = a[a[:, 2] > 0.3]
            if len(a):
                out[s]["ankle"][i] = a[:, :2].mean(0)
    return out


def ground_correct(cam, joints, ankle_px, stable_px=None):
    """Birdseye v1's correction: shift the body so VIMO's lower ankle sits where a pixel-stable 2D
    ankle ray-casts (at ankle height), MAD-reject outliers, interpolate, floor/height clamp."""
    n = len(joints)
    cams = cam if isinstance(cam, list) else [cam] * n
    c0 = next(c for c in cams if c is not None)
    stable = stable_px if stable_px is not None else STABLE_PIX_1080 * c0.size[1] / 1080.0
    shift_f, shift_v = [], []
    for f in range(1, n):
        if np.isnan(ankle_px[f]).any() or np.isnan(ankle_px[f - 1]).any() or np.isnan(joints[f, 7]).any() or cams[f] is None:
            continue
        if np.linalg.norm(ankle_px[f] - ankle_px[f - 1]) >= stable:
            continue
        g = cams[f].pixel_to_ground(ankle_px[f][None], z=ANKLE_H)[0]
        # The 2D point is the MID-ankle, so match VIMO's mid-ankle in x/y (v1 matched its lower ankle, which slid
        # the body half a stance toward the back foot: ~0.15 m toward the camera for the near player) and put
        # the lower ankle at ankle height.
        mid = joints[f, [7, 8]].mean(0)
        low_z = min(joints[f, 7, 2], joints[f, 8, 2])
        shift_f.append(f); shift_v.append(np.r_[g[:2] - mid[:2], ANKLE_H - low_z])
    shift_f, shift_v = np.array(shift_f), np.array(shift_v).reshape(-1, 3)
    if len(shift_v) >= 4:
        med = np.median(shift_v, 0)
        mad = np.median(np.abs(shift_v - med), 0) + 1e-6
        k = np.all(np.abs(shift_v - med) < MAD_K * mad, 1)
        shift_f, shift_v = shift_f[k], shift_v[k]
    full = np.zeros((n, 3)) if not len(shift_f) else np.stack(
        [np.interp(np.arange(n), shift_f, shift_v[:, c]) for c in range(3)], 1)
    out = joints + full[:, None]
    for f in range(n):
        z = out[f, [7, 8, 10, 11], 2]
        z = z[np.isfinite(z)]
        if len(z):
            m = z.min()
            out[f, :, 2] -= m if m < 0 else max(m - MAX_FOOT_H, 0)
    return out


def main():
    os.chdir(TRAM)  # TRAM loads data/smpl/... relative to its own folder
    from lib.models import get_hmr_vimo

    model = get_hmr_vimo(checkpoint=os.path.join(TRAM, "data", "pretrain", "vimo_checkpoint.pth.tar"))
    device = next(model.parameters()).device
    uids = sys.argv[1:] or sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(CLIPS, "*.mp4")))
    for uid in uids:
        dst = os.path.join(OUT, VIMO_DIR, uid + ".npz")
        cal_p, pose_p = os.path.join(OUT, CALIB_DIR, uid + ".json"), os.path.join(OUT, POSE_DIR, uid + ".npz")
        if os.path.exists(dst) or not (os.path.exists(cal_p) and os.path.exists(pose_p)):
            continue
        t0 = time.time()
        import json
        cal = json.load(open(cal_p))
        trusted = bool(cal["stats"].get("valid", False)) and cal["stats"].get("on_court_median", 1.0) > 0.8  # BirdsCourt has no on_court check
        cam = Camera.from_dict(cal["cam"]) if trusted else nominal_camera()
        video = os.path.join(CLIPS, uid + ".mp4")
        keep = native_frames(video, os.path.join(OUT, "native", uid + ".json"))
        ctp = os.path.join(OUT, "camtrack", uid + ".json")
        cams = None  # per native frame (birdseye_v2/camtrack.py): cuts, close-ups, other court cameras, pans
        if CAMTRACK and os.path.exists(ctp):
            vc = cv2.VideoCapture(video)
            raw = camtrack.load(ctp, (int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            cams = [raw[k] if k < len(raw) else None for k in keep]
            if any(c is not None for c in cams):
                trusted, cam = True, next(c for c in cams if c is not None)
            else:
                cams = None
        poses_all = load_pose(pose_p)
        poses = [poses_all[k] if k < len(poses_all) else np.zeros((0, 17, 3), np.float32) for k in keep]
        tmp = tempfile.mkdtemp(prefix="vimo_")
        try:
            cap, i, files, kset = cv2.VideoCapture(video), 0, [], set(int(k) for k in keep)
            while True:
                ok, im = cap.read()
                if not ok:
                    break
                if i in kset:
                    fn = os.path.join(tmp, f"{len(files):06d}.jpg")
                    cv2.imwrite(fn, im, [cv2.IMWRITE_JPEG_QUALITY, 95]); files.append(fn)
                i += 1
            T = len(files)
            if cams is not None:
                cams = cams[:T] + [None] * (T - len(cams[:T]))
                sb = side_boxes(cams, poses[:T])
            elif trusted:
                sb = side_boxes(cam, poses[:T])
            else:
                sys.path.insert(0, os.path.join(ROOT, "tools"))
                from fb_calibrate import court_region
                med = np.median(np.stack([cv2.imread(f) for f in files[::max(1, T // 15)]]), 0).astype(np.uint8)
                sb = side_boxes_image(poses[:T], court_region(med))
            R = cam.R
            res_out = {"frames": keep[:T], "trusted_calib": trusted, "camtrack": cams is not None}
            for s in ("top", "bottom"):
                sb[s]["box"], sb[s]["valid"] = fill_box_gaps(sb[s]["box"][:T].copy(), sb[s]["valid"][:T].copy())
                v = sb[s]["valid"][:T]
                joints = np.full((T, 24, 3), np.nan)
                if v.sum() >= 16:
                    if os.environ.get("VIMO_SLOW") == "1":  # original TRAM path (for checking fast_inference)
                        r = model.inference(np.array(files), sb[s]["box"][:T], valid=v, frame=np.arange(T),
                                            img_focal=float(cam.f), img_center=np.array(cam.size, np.float32) / 2)
                    elif cams is not None:  # one run per stretch of similar focal length (a shot / a zoom level)
                        r = per_camera_inference(model, files, sb[s]["box"][:T], v, cams)
                    else:
                        r = fast_inference(model, files, sb[s]["box"][:T], v, float(cam.f), np.array(cam.size, np.float32) / 2)
                    if r is not None:
                        so = model.smpl.query({"pred_rotmat": r["pred_rotmat"].float().to(device),
                                               "pred_shape": r["pred_shape"].float().to(device)}, default_smpl=True)
                        j_cam = (so.joints[:, :24].float().cpu() + r["pred_trans"].float()).numpy()
                        fr_ = r["frame"].numpy().astype(int)
                        if cams is not None:
                            j_w = np.stack([cams[f].R.T @ (j - cams[f].tvec).T for f, j in zip(fr_, j_cam)]).transpose(0, 2, 1)
                        else:
                            j_w = (R.T @ (j_cam - cam.tvec).transpose(0, 2, 1)).transpose(0, 2, 1)
                        joints[fr_] = j_w
                        res_out[s + "_raw"] = joints.astype(np.float32)  # before ground correction (for re-tuning)
                        res_out[s + "_ankle_px"] = sb[s]["ankle"][:T].astype(np.float32)
                        if trusted:  # needs a real floor plane
                            joints = ground_correct(cams if cams is not None else cam, joints, sb[s]["ankle"][:T])
                res_out[s] = joints.astype(np.float32)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.savez_compressed(dst, **res_out)
            print(f"{uid} {T}f top {np.isfinite(res_out['top'][:, 0, 0]).mean():.2f} "
                  f"bottom {np.isfinite(res_out['bottom'][:, 0, 0]).mean():.2f} {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            print("FAILED", uid, repr(e), flush=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
