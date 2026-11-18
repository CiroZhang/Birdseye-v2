"""Shuttle detection on the player cameras with the frozen TrackNetV3, made to see a static camera.

TrackNet takes 8 consecutive frames plus a median background of the clip, which assumes a fixed camera. The player
cameras pan and zoom, so the plain run (tools/fb_track.py) sees a smeared background and finds the shuttle in ~5% of
frames. Here every frame is processed in its own view: its 8-frame window and a background median over +-40 frames are
warped into it with the calibrated rotation + zoom homographies (outputs/cctv_<view>/pcam/calib, same camera centre,
so K_b R_b R_a^T K_a^-1 is exact up to lens distortion). Same weights, no retraining.

Output outputs/cctv_<view>/shuttle_stab/<uid>.json {frames: [k...], peaks: [[[x, y, conf], ...] per frame]} with up to
3 heatmap peaks >= PEAK per frame (full-res px), so the teacher can associate them with the other views itself.
usage: python tools/pcam_shuttle.py <view> [uid ...]   (env DEVICE=cuda|cpu, PEAK)"""
import json, os, sys, time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2 import tracknet as TNET  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True  # fp32 matmul with TF32 off stalls minutes on the RTX 5080 (NOTES.md)

W, H, SEQ = TNET.W, TNET.H, TNET.SEQ
FULL = (864, 486)
PEAK = float(os.environ.get("PEAK", "0.3"))
BG_SPAN, BG_STEP = 40, 4


def homographies(z):
    """Per frame K_t R_t at TrackNet resolution, so H_{a->b} = KR[b] @ inv(KR[a])."""
    s = W / FULL[0]
    R = Rotation.from_rotvec(z["rvec"]).as_matrix()
    K = np.zeros((len(R), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = z["f"] * s
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = W / 2, H / 2, 1
    return K @ R


def peaks(hm, thr, k=3):
    m = (hm >= thr).astype(np.uint8)
    if not m.any():
        return []
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        j = np.argmax(hm[ys, xs])
        out.append((float(xs[j]), float(ys[j]), float(hm[ys[j], xs[j]])))
    return sorted(out, key=lambda p: -p[2])[:k]


@torch.no_grad()
def run(model, view, uid, device):
    z = np.load(os.path.join(ROOT, "outputs", "cctv_" + view, "pcam", "calib", uid + ".npz"))
    frames, full = TNET.read_small_frames(os.path.join(ROOT, "data", "cctv", "clips_" + view, uid + ".mp4"))
    n = min(len(frames), len(z["f"]))
    KR = homographies(z)
    inv = np.linalg.inv(KR)
    sx, sy = full[0] / W, full[1] / H
    res = []
    batch, metas = [], []

    def flush():
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch)).to(device).float() / 255.0  # (B, 27, H, W)
        y = model(x).cpu().numpy()
        for (f, p), hm in zip(metas, y):
            res.append((f, [(px * sx, py * sy, c) for px, py, c in peaks(hm[p], PEAK)]))
        batch.clear(); metas.clear()

    for f in range(n):
        s0 = min(max(f - SEQ // 2 + 1, 0), max(n - SEQ, 0))
        win = [min(s0 + i, n - 1) for i in range(SEQ)]
        warp = lambda a: cv2.warpPerspective(frames[a], KR[f] @ inv[a], (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        seq = [frames[a] if a == f else warp(a) for a in win]
        bgs = [warp(a) if a != f else frames[a] for a in range(max(0, f - BG_SPAN), min(n, f + BG_SPAN + 1), BG_STEP)]
        bg = np.median(np.stack(bgs), 0).astype(np.uint8)
        batch.append(np.concatenate([bg] + seq, 2).transpose(2, 0, 1))
        metas.append((f, win.index(f)))
        if len(batch) == int(os.environ.get("BATCH", "16")):
            flush()
    flush()
    res.sort()
    return {"frames": [r[0] for r in res], "peaks": [[[round(a, 1), round(b, 1), round(c, 3)] for a, b, c in r[1]] for r in res]}


def main():
    view = sys.argv[1]
    uids = sys.argv[2:] or open(os.path.join(ROOT, "outputs", "cctv_pcam_dev_uids.txt")).read().split()
    device = os.environ.get("DEVICE", "cuda")
    model = TNET.load_model(device)
    od = os.path.join(ROOT, "outputs", "cctv_" + view, os.environ.get("OUT_SUB", "shuttle_stab"))
    os.makedirs(od, exist_ok=True)
    for u in uids:
        dst = os.path.join(od, u + ".json")
        if os.path.exists(dst) or not os.path.exists(os.path.join(ROOT, "outputs", "cctv_" + view, "pcam", "calib", u + ".npz")):
            continue
        t0 = time.time()
        try:
            r = run(model, view, u, device)
        except Exception as e:
            print("FAILED", u, repr(e)[:200], flush=True)
            continue
        json.dump(r, open(dst, "w"))
        hit = np.mean([len(p) > 0 for p in r["peaks"]])
        print(f"{u} {len(r['frames'])} frames, shuttle found in {hit:.0%}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
