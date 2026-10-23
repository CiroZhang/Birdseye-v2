"""RallyNet: one model from 2D detections (as camera rays, birdseye_v2.rally_obs) to the rally in 3D -- players'
skeletons, the shuttle, rackets and hits -- trained without 3D labels.

Tokens: per frame, one per player slot, one for the shuttle, one per racket box. Players enter as their keypoint
rays cut by the vertical plane through their feet (a view-independent first guess of the body, in the court
frame); the shuttle and rackets as rays. Blocks alternate attention over time (per token slot) and over the
entities of a frame, so a player can look at the shuttle at contact and the shuttle at both hitters.

Outputs: 17 joints per player (court frame, m), the shuttle's 3D position, each player's racket direction, and a
hit probability per player and frame.

Self-supervised losses (losses()):
  reprojection  joints / shuttle lie on their detected rays (angle, ~px at f=1000)
  physics       between hits the shuttle obeys gravity + drag (terminal velocity 6.4 m/s, measured on DroneData);
                the residual is ignored near predicted hits -- hits are where the flight model breaks ...
  contact       ... and at a hit the shuttle must be at the hitter's racket head, so hits cannot go anywhere
  racket        the racket head projects onto a detected racket box (when one is near)
  body          bone lengths constant per player, left = right, plausible proportions; joints above the floor,
                a foot on the floor; smooth motion
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .court import L, W

P, R, J = 4, 4, 17
G = 9.81
V_TERM = 6.4
K_DRAG = G / V_TERM ** 2
ANKLE_H = 0.12  # m, as players.foot_point (chosen on DroneData dev BD12-14)
UPRIGHT = 0.35  # m
GRIP_TO_HEAD = 0.47  # m, hand to racket-head centre (675 mm racket, head 290 mm, grip 80 mm)
SCALE = 5.0  # m, network coordinate unit
CENTRE = np.array([W / 2, L / 2, 0.0], np.float32)
# COCO-17 bones and left/right pairs
BONES = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16), (5, 6), (11, 12), (5, 11), (6, 12)]
SYM = [(0, 2), (1, 3), (4, 6), (5, 7), (10, 11)]
# bone length / stature (Winter): upper arm, forearm(to wrist), thigh, shin(to ankle), shoulder width, hip width, trunk
FRAC = [0.186, 0.146, 0.186, 0.146, 0.245, 0.246, 0.245, 0.246, 0.259, 0.191, 0.288, 0.288]


# ------------------------------------------------------------------ features
def _fill_smooth(v, sigma=3.0, max_gap=30):
    from .tracks import _fill, _smooth
    return _smooth(_fill(np.arange(len(v)), v, max_gap=max_gap, hold=10), sigma)


def featurize(o):
    """rally_obs arrays -> model inputs (numpy, float32). Body first guess X0: each keypoint ray cut by the vertical
    plane through the player's (smoothed) foot point, facing the camera."""
    C = o["C"].astype(np.float32)
    T = len(C)
    kd, kc = o["kp_dir"], o["kp_conf"]
    root = np.full((T, P, 2), np.nan, np.float32)
    X0 = np.zeros((T, P, J, 3), np.float32)
    view = np.zeros((T, P, 3), np.float32)
    present = np.zeros((T, P), np.float32)
    for p in range(P):
        f = o["foot"][:, p].astype(float)
        if np.isfinite(f[:, 0]).sum() < 5:
            continue
        f = _fill_smooth(f)
        root[:, p] = f
        ok_t = np.isfinite(f[:, 0]) & np.isfinite(C[:, 0])
        present[:, p] = ok_t & (kc[:, p].max(1) > 0)
        n = np.c_[f - C[:, :2], np.zeros(T)]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        foot3 = np.c_[f, np.zeros(T)]
        num = ((foot3 - C) * n).sum(1)  # (T,)
        d = kd[:, p]  # (T,J,3)
        den = (d * n[:, None]).sum(2)
        lam = num[:, None] / np.where(np.abs(den) < 1e-6, np.nan, den)
        X = C[:, None] + lam[..., None] * d
        rel = X - foot3[:, None]
        rel[kc[:, p] <= 0] = np.nan
        from .tracks import _fill
        for j in range(J):  # short gaps: interpolate the joint's offset from the feet
            rel[:, j] = _fill(np.arange(T), rel[:, j], max_gap=30, hold=10)
        X0[:, p] = np.nan_to_num(rel, nan=0.0)
        h = foot3 - C
        view[:, p] = np.nan_to_num(h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9))
    root = np.nan_to_num(root)
    sd = o["sh_dir"].astype(np.float32)
    dprev = np.zeros_like(sd); dnext = np.zeros_like(sd)
    dprev[1:] = sd[1:] - sd[:-1]; dnext[:-1] = sd[1:] - sd[:-1]
    sh_mot = np.nan_to_num(np.concatenate([dprev, dnext], 1) * 30.0)
    return {"sh_mot": np.clip(sh_mot, -3, 3), "C": np.nan_to_num(C), "cam_ok": np.isfinite(o["C"][:, 0]).astype(np.float32),
            "kp_dir": np.nan_to_num(kd).astype(np.float32), "kp_conf": kc.astype(np.float32),
            "X0": np.clip(X0, -3, 3), "root": root, "view": view, "present": present,
            "sh_dir": np.nan_to_num(o["sh_dir"]).astype(np.float32), "sh_vis": o["sh_vis"].astype(np.float32),
            "rk_dir": np.nan_to_num(o["rk_dir"]).astype(np.float32), "rk_size": o["rk_size"].astype(np.float32),
            "rk_conf": o["rk_conf"].astype(np.float32), "dt": np.float32(1.0 / float(o["fps"]))}


# ------------------------------------------------------------------ model
class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.ta = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ea = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, tmask, emask):
        B, T, E, D = x.shape
        h = self.n1(x).permute(0, 2, 1, 3).reshape(B * E, T, D)  # over time, per slot
        km = ~tmask.permute(0, 2, 1).reshape(B * E, T)
        km[km.all(1)] = False  # an empty slot attends to itself (output masked out later)
        a = self.ta(h, h, h, key_padding_mask=km, need_weights=False)[0]
        x = x + a.reshape(B, E, T, D).permute(0, 2, 1, 3)
        h = self.n2(x).reshape(B * T, E, D)  # over entities, per frame
        km = ~emask.reshape(B * T, E)
        km[km.all(1)] = False
        a = self.ea(h, h, h, key_padding_mask=km, need_weights=False)[0]
        x = x + a.reshape(B, T, E, D)
        return x + self.ff(self.n3(x))


def time_enc(T, d, device):
    t = torch.arange(T, device=device, dtype=torch.float32)[:, None]
    k = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(1e4) / d))
    pe = torch.zeros(T, d, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(t * k), torch.cos(t * k)
    return pe


class RallyNet(nn.Module):
    def __init__(self, d=256, depth=6, heads=8):
        super().__init__()
        self.d = d
        self.p_in = nn.Sequential(nn.Linear(J * 3 + J + 2 + 3 + 3, d), nn.GELU(), nn.Linear(d, d))
        self.s_in = nn.Sequential(nn.Linear(3 + 3 + 1 + 6, d), nn.GELU(), nn.Linear(d, d))
        self.r_in = nn.Sequential(nn.Linear(3 + 3 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.kind = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        self.p_out = nn.Linear(d, J * 3 + 2 + 3 + 1 + J)  # joints rel. root, root shift, racket dir, hit logit, depth along ray
        self.s_out = nn.Linear(d, 3 + 1)  # shuttle position (norm), depth-along-ray switch
        nn.init.zeros_(self.p_out.weight); nn.init.zeros_(self.p_out.bias)
        nn.init.zeros_(self.s_out.weight); nn.init.zeros_(self.s_out.bias)

    def forward(self, b):
        B, T = b["C"].shape[:2]
        dev = b["C"].device
        Cn = (b["C"] - torch.as_tensor(CENTRE, device=dev)) / SCALE
        pin = torch.cat([b["X0"].flatten(3) / 1.0, b["kp_conf"], b["root"] / SCALE - torch.as_tensor(CENTRE[:2], device=dev) / SCALE,
                         b["view"], Cn[:, :, None].expand(B, T, P, 3)], -1)
        sin = torch.cat([Cn, b["sh_dir"], b["sh_vis"][..., None], b["sh_mot"]], -1)[:, :, None]
        rin = torch.cat([Cn[:, :, None].expand(B, T, R, 3), b["rk_dir"], b["rk_size"][..., None] * 20, b["rk_conf"][..., None]], -1)
        x = torch.cat([self.p_in(pin) + self.kind[0], self.s_in(sin) + self.kind[1], self.r_in(rin) + self.kind[2]], 2)
        x = x + time_enc(T, self.d, dev)[None, :, None]
        pm = b["present"] > 0
        sm = (b["cam_ok"] > 0)[..., None]
        rm = b["rk_conf"] > 0
        emask = torch.cat([pm, sm, rm], 2)
        tmask = torch.cat([pm | (b["cam_ok"] > 0)[..., None], sm, rm | (b["cam_ok"] > 0)[..., None]], 2)
        for blk in self.blocks:
            x = blk(x, tmask, emask)
        po = self.p_out(x[:, :, :P])
        rel = po[..., :J * 3].reshape(B, T, P, J, 3) + b["X0"]
        root = torch.cat([b["root"] + po[..., J * 3:J * 3 + 2], torch.zeros_like(b["root"][..., :1])], -1)
        free = root[..., None, :] + rel
        # a seen joint stays exactly on its camera ray; the network only moves it in depth (m) from the first guess
        C4 = b["C"][:, :, None, None]
        lam0 = (b["X0"] + torch.cat([b["root"], torch.zeros_like(b["root"][..., :1])], -1)[..., None, :] - C4).norm(dim=-1)
        on_ray = C4 + (lam0 + po[..., J * 3 + 6:J * 3 + 6 + J])[..., None] * b["kp_dir"]
        seen = (b["kp_conf"] > 0).float()[..., None]
        joints = seen * on_ray + (1 - seen) * free
        u = F.normalize(po[..., J * 3 + 2:J * 3 + 5] + torch.tensor([0, 0, 1.0], device=dev), dim=-1)
        hit = po[..., J * 3 + 5] - 4.0  # prior: hits are rare
        so = self.s_out(x[:, :, P])
        # a free 3D point, pulled onto the detected ray by a soft loss (TrackNet's ~2 px jitter, differentiated
        # twice at 30 fps, would otherwise look like accelerations as large as gravity)
        shuttle = torch.as_tensor(CENTRE, device=dev) + SCALE * so[..., :3] + torch.tensor([0, 0, 1.5], device=dev)
        return {"joints": joints, "racket_u": u, "hit": hit, "shuttle": shuttle}


def fix_head(Jt):
    """Face joints (COCO 0-4) are rarely seen from behind and no loss shapes them: place them at head height above
    the neck (nose 0.22 m up, 6 cm forward of the shoulder line; eyes and ears around it). Numpy, (...,17,3)."""
    J = Jt.copy()
    ls, rs = J[..., 5, :], J[..., 6, :]
    neck = 0.5 * (ls + rs)
    side = rs - ls
    side[..., 2] = 0
    side /= np.linalg.norm(side, axis=-1, keepdims=True) + 1e-9
    fwd = np.stack([-side[..., 1], side[..., 0], np.zeros_like(side[..., 0])], -1)  # left->right x up = facing
    up = np.array([0, 0, 1.0])
    head = neck + 0.22 * up
    J[..., 0, :] = head + 0.06 * fwd
    J[..., 1, :] = head + 0.03 * fwd - 0.03 * side + 0.03 * up
    J[..., 2, :] = head + 0.03 * fwd + 0.03 * side + 0.03 * up
    J[..., 3, :] = head - 0.07 * side
    J[..., 4, :] = head + 0.07 * side
    return J


# ------------------------------------------------------------------ losses
def _angle_px(X, C, d):
    v = F.normalize(X - C, dim=-1)
    return (v - d).norm(dim=-1) * 1000.0  # ~ px at f = 1000


def racket_head(out, wrist_side=None):
    """Racket-head centre per player/frame: the wrist that is further from the body's centre line holds it."""
    j = out["joints"]
    lw, rw = j[..., 9, :], j[..., 10, :]
    mid = 0.5 * (j[..., 11, :] + j[..., 12, :])
    pick = ((lw - mid).norm(dim=-1) > (rw - mid).norm(dim=-1)).float()[..., None]
    wrist = pick * lw + (1 - pick) * rw
    return wrist + GRIP_TO_HEAD * out["racket_u"], wrist


def racket_hand(joints, f, return_vote=False):
    """One racket hand per player slot for the whole rally (numpy): 0 = left wrist, 1 = right. Each frame votes for
    the wrist whose camera ray passes closest to a detected racket box (within ~0.1 rad); without racket votes, the
    wrist held further from the hips. -> (P,) int"""
    C = f["C"][:, None]
    lw, rw = joints[..., 9, :], joints[..., 10, :]
    mid = 0.5 * (joints[..., 11, :] + joints[..., 12, :])
    pres = f["present"] > 0
    rk = f["rk_dir"] * (f["rk_conf"] > 0)[..., None]  # (T,R,3), zero = no box

    def gap(w):  # (T,P) angle (rad) from the wrist's ray to the nearest racket ray
        d = w - C
        d = d / (np.linalg.norm(d, axis=-1, keepdims=True) + 1e-9)
        a = np.linalg.norm(d[:, :, None] - rk[:, None], axis=-1)
        a[np.broadcast_to((f["rk_conf"] <= 0)[:, None], a.shape)] = np.inf
        return a.min(-1)
    gl, gr = gap(lw), gap(rw)
    near = pres & (np.fmin(gl, gr) < 0.1)
    reach = np.linalg.norm(rw - mid, axis=-1) - np.linalg.norm(lw - mid, axis=-1)
    hand = np.ones(joints.shape[1], int)
    vote = np.full(joints.shape[1], 0.5, np.float32)  # share of racket votes for the right hand (0.5: none)
    for q in range(joints.shape[1]):
        v = near[:, q]
        if v.sum() >= 5:
            vote[q] = (gr[v, q] < gl[v, q]).mean()
            hand[q] = int(vote[q] >= 0.5)
        elif np.isfinite(reach[pres[:, q], q]).any():
            hand[q] = int(np.nanmedian(reach[pres[:, q], q]) >= 0)
    return (hand, vote) if return_vote else hand


_PRIOR = None


def prior(dev):
    """The frozen pose prior (birdseye_v2.poseprior, trained on our two-drone poses), or None."""
    global _PRIOR
    import os
    if _PRIOR is None:
        path = os.path.join(os.path.dirname(__file__), "..", "outputs", "rallynet", "poseprior.pt")
        if os.environ.get("PRIOR", "1") == "1" and os.path.exists(path):
            from .poseprior import PosePrior
            _PRIOR = PosePrior().to(dev)
            _PRIOR.load_state_dict(torch.load(path, map_location=dev))
            _PRIOR.eval()
            for q in _PRIOR.parameters():
                q.requires_grad_(False)
        else:
            _PRIOR = False
    return _PRIOR or None


def losses(out, b, w=None):
    w = w or {}
    dev = b["C"].device
    dt = b["dt"][:, None, None]
    pres = b["present"]
    J3 = out["joints"]
    Cj = b["C"][:, :, None, None]
    # reprojection: body joints and the shuttle on their rays
    e = _angle_px(J3, Cj, b["kp_dir"])
    cw = b["kp_conf"] * pres[..., None]
    l_rep = (F.huber_loss(e, torch.zeros_like(e), delta=5.0, reduction="none") * cw).sum() / (cw.sum() + 1e-6)
    # body: bone lengths constant over time, left = right, proportions; floor; smoothness
    bl = torch.stack([(J3[..., a, :] - J3[..., c, :]).norm(dim=-1) for a, c in BONES], -1)  # (B,T,P,nb)
    pw = pres[..., None]
    # a bone's typical length from frames where both its ends were seen; every present frame is pulled to it
    seen = torch.stack([(b["kp_conf"][..., a] > 0) & (b["kp_conf"][..., c] > 0) for a, c in BONES], -1).float() * pw
    nseen = seen.sum(1, keepdim=True)
    med = ((bl * seen).sum(1, keepdim=True) / (nseen + 1e-6)).detach()
    has = (nseen > 5).float()  # (B,1,P,nb)
    l_bone = (((bl - med) ** 2) * pw * has).sum() / ((pw * has).sum() + 1e-6)
    m = (bl * seen).sum(1) / (nseen[:, 0] + 1e-6)  # (B,P,nb), differentiable
    hb = has[:, 0]
    l_sym = sum((((m[..., a] - m[..., c]) ** 2) * hb[..., a] * hb[..., c]).sum() for a, c in SYM) / (hb.sum() + 1e-6)
    frac = torch.tensor(FRAC, device=dev)
    stature = ((m / frac) * hb).sum(-1, keepdim=True) / (hb.sum(-1, keepdim=True) + 1e-6)
    stature = stature.detach().clamp(1.5, 2.1)  # a target height for the proportions, not optimised itself
    l_prop = (((m / stature - frac) ** 2) * hb).sum() / (hb.sum() + 1e-6)
    z = J3[..., 2]
    l_floor = (F.relu(-z) * pw).sum() / (pw.sum() + 1e-6) + (F.huber_loss(torch.minimum(z[..., 15], z[..., 16]),
                                                                           torch.full_like(z[..., 15], ANKLE_H), delta=0.05, reduction="none") * pres).sum() / (pres.sum() + 1e-6) * 10.0
    # foot contact: an ankle that is not moving is planted, at ankle height (only a moving foot may lift)
    av = (J3[:, 1:, :, [15, 16]] - J3[:, :-1, :, [15, 16]]).norm(dim=-1).detach() / dt[:, :1, :, None]  # (B,T-1,P,2) m/s
    planted = torch.sigmoid((0.6 - av) / 0.1) * pres[:, 1:, :, None]
    l_plant = (F.huber_loss(z[:, 1:, :, [15, 16]], torch.full_like(planted, ANKLE_H), delta=0.05, reduction="none") * planted).sum() / (planted.sum() + 1e-6)
    # balance: hips over the feet, shoulders over the hips (horizontally, within UPRIGHT m). Without it the cheapest
    # way to meet the body-height prior from a high camera is to slide the torso along its rays, away from the feet
    pel = 0.5 * (J3[..., 11, :2] + J3[..., 12, :2])
    sho = 0.5 * (J3[..., 5, :2] + J3[..., 6, :2])
    ank = 0.5 * (J3[..., 15, :2] + J3[..., 16, :2])
    l_up = ((F.relu((pel - ank).norm(dim=-1) - UPRIGHT) + F.relu((sho - pel).norm(dim=-1) - UPRIGHT)) * pres).sum() / (pres.sum() + 1e-6)
    acc = J3[:, 2:] - 2 * J3[:, 1:-1] + J3[:, :-2]
    aw = (pres[:, 2:] * pres[:, 1:-1] * pres[:, :-2])[..., None, None]
    l_smooth = ((acc ** 2) * aw).sum() / (aw.sum() * J * 3 + 1e-6) / (dt[:, :1, :, None, None] ** 2).mean() * 1e-4
    # shuttle physics away from hits
    s = out["shuttle"]
    ok = b["cam_ok"]
    v = (s[:, 2:] - s[:, :-2]) / (2 * dt)
    a = (s[:, 2:] - 2 * s[:, 1:-1] + s[:, :-2]) / dt ** 2
    gvec = torch.tensor([0, 0, -G], device=dev)
    res = a - (gvec - K_DRAG * v.norm(dim=-1, keepdim=True) * v)
    hp = torch.sigmoid(out["hit"]) * pres  # (B,T,P)
    hany = 1 - torch.prod(1 - hp, dim=-1)  # P(some player hits at t)
    near = F.max_pool1d(hany[:, None], 7, stride=1, padding=3)[:, 0]  # a hit within +-3 frames
    pw_ = (1 - near[:, 1:-1]) * ok[:, 2:] * ok[:, 1:-1] * ok[:, :-2]
    l_phys = (F.huber_loss(res / 10.0, torch.zeros_like(res), delta=1.0, reduction="none").sum(-1) * pw_).sum() / (pw_.sum() + 1e-6)
    sv = b["sh_vis"]
    l_srep = (F.huber_loss(_angle_px(s, b["C"], b["sh_dir"]), torch.zeros_like(sv), delta=3.0, reduction="none") * sv).sum() / (sv.sum() + 1e-6)
    l_sfloor = F.relu(-s[..., 2]).mean()
    # contact: a hit puts the shuttle at the hitter's racket head; hits are sparse
    head, wrist = racket_head({"joints": out["joints"].detach(), "racket_u": out["racket_u"]})
    dist = (s[:, :, None] - head).norm(dim=-1)  # (B,T,P)
    l_contact = (hp * F.relu(dist - 0.15)).sum() / (hp.sum() + 1e-6)
    l_sparse = hany.mean()
    # racket: head projects into a detected box near it
    ang = _angle_px(head[:, :, :, None], b["C"][:, :, None, None], b["rk_dir"][:, :, None])  # (B,T,P,R)
    tol = b["rk_size"][:, :, None] * 1000.0
    valid = (b["rk_conf"][:, :, None] > 0.25) & (ang < 2 * tol + 30)
    best = torch.where(valid, ang, torch.full_like(ang, 1e4)).min(-1).values
    has = (best < 1e4).float() * pres
    l_rk = (F.huber_loss(best.clamp(max=500), torch.zeros_like(best), delta=10.0, reduction="none") * has).sum() / (has.sum() + 1e-6) * 0.1
    du = out["racket_u"][:, 1:] - out["racket_u"][:, :-1]
    l_rks = (du ** 2).sum(-1).mean()
    # hit labels where the training set has them (BFMD train matches, FineBadminton-20K)
    l_hit = torch.zeros((), device=dev)
    if "hit_lbl" in b:
        lbl = F.max_pool1d(b["hit_lbl"].permute(0, 2, 1), 3, stride=1, padding=1).permute(0, 2, 1)  # +-1 frame
        hm = b["hit_has"][:, None, None] * pres
        bce = F.binary_cross_entropy_with_logits(out["hit"], lbl, pos_weight=torch.tensor(10.0, device=dev), reduction="none")
        l_hit = (bce * hm).sum() / (hm.sum() + 1e-6)
    if "hit_any" in b:
        la = F.max_pool1d(b["hit_any"][:, None], 3, stride=1, padding=1)[:, 0]
        ha = b["hit_any_has"][:, None] * b["cam_ok"]
        pa = hany.clamp(1e-5, 1 - 1e-5)
        bce = -(10.0 * la * torch.log(pa) + (1 - la) * torch.log(1 - pa))
        l_hit = l_hit + (bce * ha).sum() / (ha.sum() + 1e-6)
    # pose plausibility (learned prior) and, on DroneData dev rallies, the two-drone triangulated joints
    l_prior = torch.zeros((), device=dev)
    pp = prior(dev)
    if pp is not None:
        on = pres > 0
        l_prior = pp.penalty(J3[on]).mean() if on.any() else torch.zeros((), device=dev)
    l_j3d = torch.zeros((), device=dev)
    if "J3d_mask" in b:
        jm = b["J3d_mask"]
        l_j3d = (F.huber_loss(J3, b["J3d"], delta=0.1, reduction="none").sum(-1) * jm).sum() / (jm.sum() + 1e-6)
    parts = {"plant": l_plant * 100, "prior": l_prior * 20, "j3d": l_j3d * 20, "hit": l_hit * 5, "up": l_up * 20, "rep": l_rep, "bone": l_bone * 100, "sym": l_sym * 100, "prop": l_prop * 30, "floor": l_floor * 10,
             "smooth": l_smooth, "phys": l_phys, "srep": l_srep, "sfloor": l_sfloor * 10, "contact": l_contact * 10,
             "sparse": l_sparse * 1, "rk": l_rk, "rks": l_rks}
    parts = {k: v * w.get(k, 1.0) for k, v in parts.items()}
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}
