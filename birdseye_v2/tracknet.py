"""Lean GPU inference for the pretrained TrackNetV3 shuttle detector (third_party/TrackNetV3).

Same model, input format (median background + 8 RGB frames at 512x288, /255) and positional
temporal ensemble as the official predict.py, but everything after decoding stays on the GPU.
The official script spends minutes per 4K clip in CPU-side PIL resizing and tensor shuffling.
InpaintNet is deliberately NOT used: on frames with no detection it emits the same canned
trajectory for every video.

Output per frame: visibility, x, y (full-res px, centre of the largest >0.5 heatmap blob, as
in TrackNetV3), and conf = ensembled heatmap peak."""
import math
import os
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TN = os.path.join(ROOT, "third_party", "TrackNetV3")
W, H, SEQ = 512, 288, 8

torch.backends.cudnn.allow_tf32 = False  # avoids a ~40s lazy cuDNN JIT stall on this GPU


def load_model(device="cuda"):
    sys.path.insert(0, TN)
    from model import TrackNet  # noqa: E402

    ck = torch.load(os.path.join(TN, "ckpts", "TrackNet_best.pt"), map_location=device)
    p = ck["param_dict"]
    assert p["bg_mode"] == "concat" and p["seq_len"] == SEQ
    m = TrackNet(in_dim=(SEQ + 1) * 3, out_dim=SEQ).to(device).eval()
    m.load_state_dict(ck["model"])
    return m


def _ensemble_weight():
    w = torch.ones(SEQ)
    for i in range(math.ceil(SEQ / 2)):
        w[i] = i + 1
        w[SEQ - i - 1] = i + 1
    return w / w.sum()


def read_small_frames(path):
    cap = cv2.VideoCapture(path)
    full = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    frames = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1])
    return np.stack(frames), full


@torch.no_grad()
def detect(model, frames_rgb, full_size, batch=16, device="cuda", chunk=1500):
    n = len(frames_rgb)
    if n > chunk:  # whole clip on the GPU is ~1.8 MB/frame (+0.6 for heatmaps); long FBBench clips spill past 16 GB
        out = []
        for a in range(0, n, chunk):
            lo, hi = max(a - SEQ, 0), min(a + chunk + SEQ, n)  # SEQ frames of context on each side
            rows = detect(model, frames_rgb[lo:hi], full_size, batch, device, chunk=hi - lo)
            out += [(f + lo, *r) for f, *r in rows if a <= f + lo < a + chunk]
        return out
    x =torch.from_numpy(np.ascontiguousarray(frames_rgb)).to(device).permute(0, 3, 1, 2).float() / 255.0
    median = x.median(0).values  # (3,H,W)
    starts = list(range(0, max(n - SEQ + 1, 1)))
    acc = torch.zeros(n, H, W, device=device)
    wsum = torch.zeros(n, device=device)
    wt = _ensemble_weight().to(device)
    for b0 in range(0, len(starts), batch):
        ss = starts[b0:b0 + batch]
        idx = torch.tensor([[min(s + k, n - 1) for k in range(SEQ)] for s in ss], device=device)
        inp = torch.cat([median.expand(len(ss), 3, H, W), x[idx].reshape(len(ss), SEQ * 3, H, W)], 1)
        y = model(inp)  # (B,SEQ,H,W) sigmoid heatmaps
        for j, s in enumerate(ss):
            for k in range(SEQ):
                f = s + k
                if f < n:
                    acc[f] += wt[k] * y[j, k]
                    wsum[f] += wt[k]
    heat = (acc / wsum.clamp(min=1e-6)[:, None, None]).cpu().numpy()
    sx, sy = full_size[0] / W, full_size[1] / H
    out = []
    for f in range(n):
        hm = heat[f]
        conf = float(hm.max())
        mask = (hm > 0.5).astype(np.uint8)
        if mask.any():
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x0, y0, w0, h0 = max((cv2.boundingRect(c) for c in cnts), key=lambda r: r[2] * r[3])
            out.append((f, 1, (x0 + w0 / 2) * sx, (y0 + h0 / 2) * sy, conf))
        else:
            out.append((f, 0, 0.0, 0.0, conf))
    return out


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("Frame,Visibility,X,Y,Conf\n")
        for f, v, x, y, c in rows:
            fh.write(f"{f},{v},{x:.1f},{y:.1f},{c:.3f}\n")
