"""HitNet: who hits the shuttle when, from where the shuttle is relative to each player in the image.

A hit is the shuttle reaching a player's racket and leaving it on a new course. RallyPhys sees the same detections
but only as absolute camera rays mixed with everything else; here every player's frame is described relative to the
shuttle, in the image plane around the shuttle's ray and in units of that player's image height:
  where the shuttle is from the player's wrists, elbows, head, hips, feet and nearest racket,
  how the shuttle moves just before and just after (angular velocity, in the same local image axes),
  the player's own 2D pose (the swing).
A small transformer mixes time and the players (only one of them hits). It never sees 3D, so its hits do not
depend on the reconstruction. Outputs a hit logit per frame and player (T,P), as RallyPhys's hit head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rallynet import P, R, J

PARTS = [9, 10, 7, 8, 0, 11, 12, 15, 16]  # wrists, elbows, nose, hips, ankles
NF = len(PARTS) * 3 + 3 + J * 2 + 3 + 9  # per player-frame features (see feats)


def _unit(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-9)


def feats(b):
    """(B,T,P,NF) relative features and (B,T,P) mask."""
    s = _unit(b["sh_dir"])  # (B,T,3) shuttle ray
    vis = b["sh_vis"] > 0
    up = torch.tensor([0.0, 0.0, 1.0], device=s.device).expand_as(s)
    h = _unit(torch.cross(s, up, dim=-1))  # image-right at the shuttle
    v = torch.cross(h, s, dim=-1)  # image-up

    def loc(d):  # (B,T,...,3) ray differences -> local image-plane 2D (rad)
        sh = h.shape[:2] + (1,) * (d.dim() - 3) + (3,)
        return torch.stack([(d * h.reshape(sh)).sum(-1), (d * v.reshape(sh)).sum(-1)], -1)
    kd, kc = _unit(b["kp_dir"]), b["kp_conf"]  # (B,T,P,J,3), (B,T,P,J)
    ok = kc > 0
    # the player's image height (rad): head to feet
    feet = _unit(kd[..., 15, :] + kd[..., 16, :])
    ht = (kd[..., 0, :] - feet).norm(dim=-1).clamp(min=0.01)  # (B,T,P)
    rel = loc(kd - s[:, :, None, None]) / ht[..., None, None]  # (B,T,P,J,2) keypoint minus shuttle, in heights
    part = torch.cat([rel[..., PARTS, :].clamp(-6, 6), ok[..., PARTS, None].float()], -1)  # (B,T,P,9,3)
    part = torch.where((vis[:, :, None, None, None] & ok[..., PARTS, None]), part, torch.zeros_like(part))
    dw = rel[..., [9, 10], :].norm(dim=-1)  # shuttle to each wrist
    dw = torch.where(ok[..., [9, 10]] & vis[:, :, None, None], dw, torch.full_like(dw, 6.0)).clamp(max=6.0)
    # nearest racket to this player's wrists, and where the shuttle is from it
    rk = _unit(b["rk_dir"])  # (B,T,R,3)
    rv = b["rk_conf"] > 0.2
    wr = kd[..., [9, 10], :]  # (B,T,P,2,3)
    dwr = (loc(rk[:, :, None, None] - wr[..., None, :]).norm(dim=-1) / ht[..., None, None])  # (B,T,P,2,R)
    dwr = torch.where(rv[:, :, None, None] & ok[..., [9, 10], None], dwr, torch.full_like(dwr, 9.0)).amin(3)  # (B,T,P,R)
    near, idx = dwr.min(-1)  # (B,T,P)
    rk_n = torch.gather(rk[:, :, None].expand(-1, -1, P, -1, -1), 3, idx[..., None, None].expand(-1, -1, -1, 1, 3))[..., 0, :]
    has = (near < 1.0) & vis[:, :, None]
    rsh = torch.where(has[..., None], (loc(rk_n - s[:, :, None]) / ht[..., None]).clamp(-6, 6), torch.zeros_like(near[..., None].expand(-1, -1, -1, 2)))
    rkf = torch.cat([rsh, has[..., None].float()], -1)  # (B,T,P,3)
    # the player's own 2D pose: keypoints from the hip centre, in heights
    hip = _unit(kd[..., 11, :] + kd[..., 12, :])
    pose = torch.where(ok[..., None], (loc(kd - hip[..., None, :]) / ht[..., None, None]).clamp(-3, 3), torch.zeros_like(rel))
    # the shuttle's angular velocity before / after, in the local axes (rad per player height per second)
    dt = b["dt"].reshape(-1, 1, 1)
    vprev = torch.zeros_like(s); vnext = torch.zeros_like(s)
    vprev[:, 1:] = s[:, 1:] - s[:, :-1]; vnext[:, :-1] = s[:, 1:] - s[:, :-1]
    vok = vis.clone(); vok[:, 1:] &= vis[:, :-1]
    nok = vis.clone(); nok[:, :-1] &= vis[:, 1:]
    vp = torch.where(vok[..., None], loc(vprev) / dt, torch.zeros_like(s[..., :2]))
    vn = torch.where(nok[..., None], loc(vnext) / dt, torch.zeros_like(s[..., :2]))
    shf = torch.cat([vp, vn, vn - vp, vis[..., None].float(), vok[..., None].float(), nok[..., None].float()], -1)  # (B,T,9)
    shf = (shf[:, :, None] / torch.cat([ht[..., None].expand(-1, -1, -1, 6), torch.ones_like(ht[..., None]).expand(-1, -1, -1, 3)], -1)).clamp(-30, 30) / 10
    x = torch.cat([part.flatten(-2), dw, torch.log(ht)[..., None], pose.flatten(-2), rkf, shf], -1)
    return x, b["present"] > 0


class Block(nn.Module):
    def __init__(self, d, heads=4):
        super().__init__()
        self.t = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True)
        self.p = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True)

    def forward(self, x, mask):  # x (B,T,P,d); mask (B,T,P) present
        B, T, Pn, d = x.shape
        xt = self.t(x.permute(0, 2, 1, 3).reshape(B * Pn, T, d)).reshape(B, Pn, T, d).permute(0, 2, 1, 3)
        pad = ~mask.reshape(B * T, Pn)
        pad = pad & ~pad.all(-1, keepdim=True)  # a frame with nobody: attend anyway (output masked later)
        return self.p(xt.reshape(B * T, Pn, d), src_key_padding_mask=pad).reshape(B, T, Pn, d)


class HitNet(nn.Module):
    def __init__(self, d=128, layers=4, max_t=512):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(NF, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, max_t, 1, d))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(d) for _ in range(layers)])
        self.out = nn.Linear(d, 1)

    def forward(self, b):
        x, m = feats(b)
        x = torch.nan_to_num(x)
        T = x.shape[1]
        h = self.inp(x) + self.pos[:, :T]
        for blk in self.blocks:
            h = blk(h, m)
        return self.out(h)[..., 0] - 4.0  # (B,T,P) hit logit (prior: hits are rare)


def loss(logit, b, sigma=1.0):
    """Per-player hit labels where a set has them (BFMD / FineBadminton); 'some player hit here' where only the
    frame is labelled (DroneData training matches). Targets are soft in time (a label is +-1 frame)."""
    pres = b["present"]
    p = torch.sigmoid(logit) * pres
    T = logit.shape[1]
    k = torch.arange(-3, 4, device=logit.device, dtype=logit.dtype)
    ker = torch.exp(-0.5 * (k / sigma) ** 2)

    def soften(y):  # (B,T,...) max-spread of the labels in time
        yt = y.transpose(1, -1).reshape(-1, 1, T)
        z = (F.pad(yt, (3, 3)).unfold(-1, 7, 1) * ker).amax(-1)
        return z.reshape(y.transpose(1, -1).shape).transpose(1, -1)
    lp = torch.zeros((), device=logit.device)
    hp = b["hit_has"] > 0
    if hp.any():
        y = soften(b["hit_lbl"][hp])
        w = pres[hp] * (1 + 9 * y)
        lp = (F.binary_cross_entropy_with_logits(logit[hp], y, reduction="none") * w).sum() / (w.sum() + 1e-6)
    la = torch.zeros((), device=logit.device)
    ha = b["hit_any_has"] > 0
    if ha.any():
        y = soften(b["hit_any"][ha])
        pa = (1 - torch.prod(1 - p[ha], -1)).clamp(1e-5, 1 - 1e-5)
        w = 1 + 9 * y
        la = (F.binary_cross_entropy(pa, y, reduction="none") * w).sum() / (w.sum() + 1e-6)
    return lp + la, {"hit_p": lp.item(), "hit_any": la.item()}
