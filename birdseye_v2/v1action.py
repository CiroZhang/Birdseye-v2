"""Birdseye v1's hit + stroke-type model (3:Action, HitSpotterTransformer, 92.7% hit F1 / ~78% stroke
accuracy on BFMD) applied to VIMO skeletons from tools/fb_vimo.py.

Same input as v1's training (build_training_data_selfdetected.py / train_final.py): SMPL-24 joints,
pelvis-relative, the top player rotated 180 deg about z, frame-difference velocity and acceleration,
opponent context, 30 fps; mirror test-time ensemble; peaks at threshold 0.5, >= 10 frames apart.
FBBench footage is 25 fps, so skeletons are resampled to 30 fps first and peaks mapped back."""
import os, sys

import numpy as np
import torch

_V1 = os.path.join(os.path.dirname(__file__), "..", "third_party", "birdseye_v1", "action")
sys.path.insert(0, os.path.abspath(_V1))
from hit_model_transformer import HitSpotterTransformer  # noqa: E402

# build_training_data_selfdetected.py: shot_type_vocab = sorted(all types)
V1_VOCAB = ["block", "clear", "drive", "drop", "flick_serve", "lift", "net_kill", "net_shot", "press", "push",
            "serve", "smash"]
TO_FB = {"block": "Block", "clear": "Clear", "drive": "Drive", "drop": "Drop shot", "flick_serve": "Serve",
         "lift": "Net lift", "net_kill": "Net kill", "net_shot": "Net shot", "press": "Push shot", "push": "Push shot",
         "serve": "Serve", "smash": "Smash"}
LR_SWAP = {1: 2, 2: 1, 4: 5, 5: 4, 7: 8, 8: 7, 10: 11, 11: 10, 13: 14, 14: 13,
           16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20, 22: 23, 23: 22}
LR_PERM = [LR_SWAP.get(i, i) for i in range(24)]
THRESH, MIN_GAP = 0.5, 10


def load(device="cuda"):
    m = HitSpotterTransformer(in_dim=360, d_model=192, n_heads=6, n_layers=4, n_shot_types=12)
    m.load_state_dict(torch.load(os.path.join(_V1, "hit_model_final_transformer.pt"), map_location="cpu"))
    return m.to(device).eval()


def _resample(j, src_fps, dst_fps):
    T = len(j)
    t_dst = np.arange(0, (T - 1) / src_fps + 1e-9, 1 / dst_fps)
    t_src = np.arange(T) / src_fps
    flat = j.reshape(T, -1)
    out = np.stack([np.interp(t_dst, t_src, flat[:, c]) for c in range(flat.shape[1])], 1)
    return out.reshape(len(t_dst), 24, 3), t_dst


def _canon(j, side):
    rel = j - j[:, :1]
    if side == "top":
        rel = rel.copy(); rel[..., 0] *= -1; rel[..., 1] *= -1
    return rel


def _vel(r):
    v = np.zeros_like(r); v[1:] = r[1:] - r[:-1]
    return v


def _mirror(r):
    m = r[:, LR_PERM].copy(); m[..., 0] *= -1
    return m


def _feats(s_rel, o_rel):
    sv, ov = _vel(s_rel), _vel(o_rel)
    sa = _vel(sv)
    T = len(s_rel)
    return np.concatenate([s_rel.reshape(T, -1), sv.reshape(T, -1), sa.reshape(T, -1),
                           o_rel.reshape(T, -1), ov.reshape(T, -1)], 1).astype(np.float32)


def _interp_nan(j):
    """Fill missing frames (NaN) by interpolation; v1 trained with nan_to_num on mostly-complete data."""
    j = j.copy()
    ok = np.isfinite(j[:, 0, 0])
    if ok.sum() < 2:
        return np.nan_to_num(j)
    idx = np.arange(len(j))
    flat = j.reshape(len(j), -1)
    for c in range(flat.shape[1]):
        flat[:, c] = np.interp(idx, idx[ok], flat[ok, c])
    return flat.reshape(j.shape)


def predict(model, top, bottom, src_fps=25.0, device="cuda"):
    """top, bottom: (T,24,3) world joints at src_fps. Returns per side: hit_prob (T_src,),
    shot_prob (T_src, 12) and peaks [(frame_src, prob, fb_type, type_probs)]."""
    out = {}
    jt, t30 = _resample(_interp_nan(top), src_fps, 30.0)
    jb, _ = _resample(_interp_nan(bottom), src_fps, 30.0)
    rel = {"top": _canon(jt, "top"), "bottom": _canon(jb, "bottom")}
    for s, o in (("top", "bottom"), ("bottom", "top")):
        x = torch.tensor(_feats(rel[s], rel[o]))[None].to(device)
        xm = torch.tensor(_feats(_mirror(rel[s]), _mirror(rel[o])))[None].to(device)
        with torch.no_grad():
            hl, sl = model(x)
            hlm, slm = model(xm)
        hp = ((torch.sigmoid(hl) + torch.sigmoid(hlm)) / 2)[0].cpu().numpy()
        sp = torch.softmax(sl, -1)[0].cpu().numpy()
        peaks, k = [], np.argsort(-hp)
        for i in k:
            if hp[i] < THRESH:
                break
            if all(abs(i - p) >= MIN_GAP for p in peaks):
                peaks.append(int(i))
        peaks.sort()
        T = len(top)
        src_t = np.arange(T) / src_fps
        out[s] = {"hit_prob": np.interp(src_t, t30, hp), "shot_prob": np.stack([np.interp(src_t, t30, sp[:, c]) for c in range(12)], 1),
                  "peaks": [(float(t30[p] * src_fps), float(hp[p]), TO_FB[V1_VOCAB[int(sp[p].argmax())]], sp[p].tolist())
                            for p in peaks]}
    return out
