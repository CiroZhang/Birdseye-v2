"""RallyPhys: the rally as a small mechanical system whose forces a transformer predicts.

One principle replaces the post-hoc physics layer and most of RallyNet's hand-made losses: every entity moves
under gravity, and any other force must come from a contact. The network explains the detections (camera rays,
birdseye_v2.rally_obs) with as little unexplained force as it can:

  shuttle   a chain of flights. At a hit the network launches the shuttle from a point near the hitter's racket
            head (on the shuttle's ray when it is seen); gravity + drag (integrated inside the model) carry it to
            the next hit, where it must arrive at the next launch point. Smooth flight is by construction, depth
            comes from the rackets. (A hard launch at the racket head failed: the head is too rough a guess, no
            velocity fits the rays from there and the network learned to barely launch at all.)
            (Tried first: free impulses that are only cheap near a racket -- a soft contact cost did not pin depth
            down; the shuttle drifted metres along its rays. outputs/rallyphys_impulse_version.py.bak)
  player    a skeleton with bone lengths fixed per player (forward kinematics from bone directions). The pelvis
            needs a supporting force to stay up, and support only exists while a foot is on the floor: players
            stand on the floor because otherwise they would fall. That is what fixes depth along a single camera's
            rays. Joints move with little effort (robust acceleration cost).
  racket    a rigid stick in one hand (the network picks the hand once per player, helped by which wrist the racket
            detections sit next to), held in the forearm's own frame; the wrist turns it with little effort.

Training energy  E = E_obs (detections on their rays) + E_force (the above) + E_prior (learned pose prior from our
two-drone triangulation; hit labels where a dataset has them). No 3D labels, no hand-made rules about poses.

Outputs match RallyNet's (joints (T,P,17,3), racket_u, hit logit (T,P), shuttle (T,3)), plus the hit frames used."""
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import rallynet as RN
from .rallynet import ANKLE_H, CENTRE, GRIP_TO_HEAD, J, K_DRAG, P, R, SCALE, G, Block, time_enc, _angle_px

# skeleton: bone directions predicted per frame, lengths per player (m, for a 1.75 m adult; Winter's proportions)
#  0 hip axis (pelvis->left hip; right = -)   1 l thigh  2 l shin  3 r thigh  4 r shin   5 trunk (pelvis->neck)
#  6 shoulder axis (neck->left shoulder)      7 l upper arm  8 l forearm  9 r upper arm  10 r forearm  11 head
NB = 12
LEN0 = [0.167, 0.43, 0.43, 0.50, 0.227, 0.326, 0.256, 0.22]  # hip half, thigh, shin, trunk, shoulder half, upper, fore, head
BONE_LEN = [0, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6, 7]  # which length each bone uses (left = right by construction)
# the first-guess direction of each bone from the flat body X0: (from, to) joints, or None
_X0_BONES = [(12, 11), (11, 13), (13, 15), (12, 14), (14, 16), None, (6, 5), (5, 7), (7, 9), (6, 8), (8, 10), None]
_DOWN = [1, 2, 3, 4, 7, 8, 9, 10]
SUB = 2  # integration substeps per frame
HIT_THR = 0.5
SHUTTLE_TRAIN = os.environ.get("SHUTTLE_TRAIN", "0") == "1"
VTEACH = float(os.environ.get("VTEACH", "30"))  # VIMO bone-direction teacher weight (tools/train_rallynet.vimo_teacher)


def _unit(v, eps=1e-6):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


class RallyPhys(nn.Module):
    def __init__(self, d=256, depth=6, heads=8):
        super().__init__()
        self.d = d
        self.p_in = nn.Sequential(nn.Linear(J * 3 + J + 2 + 3 + 3 + 1, d), nn.GELU(), nn.Linear(d, d))
        self.s_in = nn.Sequential(nn.Linear(3 + 3 + 1 + 6, d), nn.GELU(), nn.Linear(d, d))
        self.r_in = nn.Sequential(nn.Linear(3 + 3 + 2, d), nn.GELU(), nn.Linear(d, d))
        self.kind = nn.Parameter(torch.randn(3, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(depth))
        # per player and frame: pelvis shift (3), bone directions (NB*3), face offsets (4*3), racket dir in each forearm's frame (2*3)
        self.p_out = nn.Linear(d, 3 + NB * 3 + 12 + 6 + 1)
        self.len_out = nn.Linear(d, len(LEN0))  # per player, pooled over time
        self.hand_out = nn.Linear(d, 1)
        # shuttle, per frame: the contact point if this frame is a hit -- depth along the ray (1), offset when unseen (3)
        self.s_out = nn.Linear(d, 4)
        for m in (self.p_out, self.len_out, self.hand_out, self.s_out):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, b):
        B, T = b["C"].shape[:2]
        dev = b["C"].device
        Cn = (b["C"] - torch.as_tensor(CENTRE, device=dev)) / SCALE
        pin = torch.cat([b["X0"].flatten(3), b["kp_conf"], b["root"] / SCALE - torch.as_tensor(CENTRE[:2], device=dev) / SCALE,
                         b["view"], Cn[:, :, None].expand(B, T, P, 3), b["hand"][..., None]], -1)
        sin = torch.cat([Cn, b["sh_dir"], b["sh_vis"][..., None], b["sh_mot"]], -1)[:, :, None]
        rin = torch.cat([Cn[:, :, None].expand(B, T, R, 3), b["rk_dir"], b["rk_size"][..., None] * 20, b["rk_conf"][..., None]], -1)
        x = torch.cat([self.p_in(pin) + self.kind[0], self.s_in(sin) + self.kind[1], self.r_in(rin) + self.kind[2]], 2)
        x = x + time_enc(T, self.d, dev)[None, :, None]
        pm = b["present"] > 0
        sm = (b["cam_ok"] > 0)[..., None]
        rm = b["rk_conf"] > 0
        emask = torch.cat([pm, sm, rm], 2)
        tmask = torch.cat([pm | sm, sm, rm | sm], 2)
        for blk in self.blocks:
            x = blk(x, tmask, emask)
        return x

    # -------------------------------------------------------------- players
    def body(self, b, xp):
        B, T = b["C"].shape[:2]
        dev = b["C"].device
        po = self.p_out(xp)  # (B,T,P,.)
        X0, kc = b["X0"], b["kp_conf"]
        up = torch.tensor([0, 0, 1.0], device=dev)
        side = torch.stack([-b["view"][..., 1], b["view"][..., 0], torch.zeros_like(b["view"][..., 0])], -1)
        side = _unit(side)
        seen = kc > 0
        both = lambda a, c: (seen[..., a] & seen[..., c])[..., None]
        hip0 = 0.5 * (X0[..., 11, :] + X0[..., 12, :])
        sho0 = 0.5 * (X0[..., 5, :] + X0[..., 6, :])
        d0 = []
        for i, bc in enumerate(_X0_BONES):
            dflt = -up if i in _DOWN else (side if i in (0, 6) else up)
            dflt = dflt.expand(B, T, P, 3)
            if bc is not None:
                v, ok = X0[..., bc[1], :] - X0[..., bc[0], :], both(*bc)
            elif i == 5:
                v, ok = sho0 - hip0, both(11, 12) & both(5, 6)
            else:
                v, ok = X0[..., 0, :] - sho0, both(5, 6) & seen[..., 0:1]
            ok = ok & (v.norm(dim=-1, keepdim=True) > 0.05)
            d0.append(torch.where(ok, _unit(v), dflt))
        d0 = torch.stack(d0, -2)  # (B,T,P,NB,3)
        D = _unit(d0 + po[..., 3:3 + NB * 3].reshape(B, T, P, NB, 3))
        # one set of bone lengths per player for the window
        pm = b["present"][..., None]
        pooled = (xp * pm).sum(1) / (pm.sum(1) + 1e-6)  # (B,P,d)
        Lp = torch.tensor(LEN0, device=dev) * torch.exp(0.3 * torch.tanh(self.len_out(pooled)))  # (B,P,8)
        Lb = Lp[:, None, :, BONE_LEN, None]  # (B,1,P,NB,1)
        V = D * Lb
        root3 = torch.cat([b["root"], torch.zeros_like(b["root"][..., :1])], -1)
        pel0 = torch.where(both(11, 12), hip0, torch.tensor([0, 0, 0.95], device=dev).expand_as(hip0))
        pel = root3 + pel0 + po[..., :3]
        lhip, rhip = pel + V[..., 0, :], pel - V[..., 0, :]
        lkn, rkn = lhip + V[..., 1, :], rhip + V[..., 3, :]
        lan, ran = lkn + V[..., 2, :], rkn + V[..., 4, :]
        neck = pel + V[..., 5, :]
        lsh, rsh = neck + V[..., 6, :], neck - V[..., 6, :]
        lel, rel = lsh + V[..., 7, :], rsh + V[..., 9, :]
        lwr, rwr = lel + V[..., 8, :], rel + V[..., 10, :]
        nose = neck + V[..., 11, :]
        face = nose[..., None, :] + 0.1 * torch.tanh(po[..., 3 + NB * 3:3 + NB * 3 + 12].reshape(B, T, P, 4, 3))
        joints = torch.stack([nose, face[..., 0, :], face[..., 1, :], face[..., 2, :], face[..., 3, :], lsh, rsh, lel, rel,
                              lwr, rwr, lhip, rhip, lkn, rkn, lan, ran], -2)
        # racket: a rigid stick in one hand, its direction given in that forearm's frame (a: along the forearm,
        # n: elbow-plane normal). Which hand is the network's call, one per player for the window (pi = P(right)).
        us, ws = [], []
        for k, (el, wr, sh) in enumerate([(lel, lwr, lsh), (rel, rwr, rsh)]):
            a = _unit(wr - el)
            ref = _unit(sh - el) + 0.3 * D[..., 6, :]
            n = _unit(torch.cross(a, ref, dim=-1))
            c = torch.cross(n, a, dim=-1)
            ul = po[..., -7 + 3 * k:po.shape[-1] - 4 + 3 * k] + torch.tensor([1.0, 0, 0], device=dev)
            us.append(_unit(ul[..., :1] * a + ul[..., 1:2] * n + ul[..., 2:3] * c))
            ws.append(wr)
        u2, w2 = torch.stack(us, -2), torch.stack(ws, -2)  # (B,T,P,2,3)
        pi = torch.sigmoid(self.hand_out(pooled)[..., 0] + 4.0 * (b["hand"][:, 0] - 0.5))  # (B,P)
        return joints, u2, w2, pi, Lp, pel, po[..., -1] - 4.0  # hit logit per player (prior: hits are rare)

    # -------------------------------------------------------------- shuttle
    def knots(self, b, xs, racket):
        """Where the shuttle is at every frame IF that frame is a contact (hit) or a window edge: on the shuttle's
        ray at about the hitter's racket depth when it is seen, else near the racket head. (B,T,3)"""
        so = self.s_out(xs)
        lam = (racket - b["C"]).norm(dim=-1, keepdim=True).clamp(min=1.0) * torch.exp(torch.tanh(so[..., :1]))
        on_ray = b["C"] + lam * b["sh_dir"]
        return torch.where((b["sh_vis"] > 0)[..., None], on_ray, racket + 0.5 * torch.tanh(so[..., 1:4]))

    def forward(self, b, hits=None, has=None):
        """hits: (B,T) bool hit frames to chain the flights at (training: the labels, for the rallies has (B,) marks);
        otherwise the network's own (peaks of its hit probability)."""
        x = self.encode(b)
        joints, u2, w2, pi, Lp, pel, hl = self.body(b, x[:, :, :P])
        right = (pi > 0.5)[:, None, :, None].expand_as(u2[..., 0, :])
        u, wr = torch.where(right, u2[..., 1, :], u2[..., 0, :]), torch.where(right, w2[..., 1, :], w2[..., 0, :])
        head = wr + GRIP_TO_HEAD * u
        pres = b["present"] > 0
        hl = torch.where(pres, hl, torch.full_like(hl, -20.0))
        gate = 1 - torch.prod(1 - torch.sigmoid(hl), -1)  # P(some player hits at t)
        g = gate.detach()
        own = (g > HIT_THR) & (g >= F.max_pool1d(g[:, None], 13, stride=1, padding=6)[:, 0])
        hits = own if hits is None else torch.where(has[:, None], hits > 0, own)
        # the hitter's racket head: the most likely player's (straight-through), body not moved by the shuttle
        soft = torch.softmax(hl, -1)
        resp = (F.one_hot(soft.argmax(-1), P).float() + soft - soft.detach())[..., None]
        racket = ((wr.detach() + GRIP_TO_HEAD * u) * resp).sum(2)
        K = self.knots(b, x[:, :, P], racket)
        # the shuttle itself is solved at inference (rally_shuttle: exact flights between contacts, fitted to the
        # rays); training it through the network only added noise to the racket (SHUTTLE_TRAIN=1 to try again)
        S = assemble(K, hits, b["dt"]) if SHUTTLE_TRAIN or not self.training else K
        return {"joints": joints, "racket_u": u, "hit": hl, "shuttle": S, "knot": K, "racket": racket,
                "hits": hits.float(), "gate": gate, "len": Lp, "pel": pel, "head": head, "u2": u2, "w2": w2, "pi": pi,
                "hand_p": pi[:, None].expand(-1, S.shape[1], -1)}


def _sim(p, v, h, N, h0=None, h1=None):
    """Gravity + drag from (p, v). h (M,1) substep for one frame / SUB. h0 (M,1): an optional first partial frame
    (the flight starts between frames); N whole frames follow; h1 (M,1): an optional last partial frame (it ends
    between frames). -> (M,N+1,3): positions after the partial start and after each whole frame (+ end if h1)."""
    gv = torch.tensor([0, 0, -G], device=p.device)

    def step(p, v, hh):
        for _ in range(SUB):
            v = v + hh * (gv - K_DRAG * v.norm(dim=-1, keepdim=True) * v)
            p = p + hh * v
        return p, v
    if h0 is not None:
        p, v = step(p, v, h0)
    out = [p]
    for _ in range(N):
        p, v = step(p, v, h)
        out.append(p)
    if h1 is not None:
        p, v = step(p, v, h1)
        out.append(p)
    return torch.stack(out, 1)


def flights(pa, pb, ta, tb, dt, iters=6):
    """The one gravity + drag flight from pa at time ta to pb at time tb (frames, fractional: a hit happens between
    frames) -- a two-point boundary value problem: Newton on the launch velocity (finite-difference Jacobian, no
    grad), then one differentiable Newton step so gradients reach the endpoints (and times) as the implicit function
    theorem gives them. Returns positions at the whole frames ceil(ta) .. floor(tb): (M,N+1,3), and those frames' count
    n (M,) and first frame f0 (M,)."""
    M = len(pa)
    f0 = torch.ceil(ta - 1e-6)
    f1 = torch.floor(tb + 1e-6)
    n = (f1 - f0).long().clamp(min=0)
    N = int(n.max())
    h = (dt / SUB)[:, None]
    h0 = ((f0 - ta) * dt / SUB)[:, None]
    h1 = ((tb - f1) * dt / SUB)[:, None]
    tf = ((tb - ta) * dt)[:, None].clamp(min=1e-3)
    ar = torch.arange(M, device=pa.device)
    gv = torch.tensor([0, 0, -G], device=pa.device)
    eps = 0.05

    with torch.no_grad():
        p0, p1 = pa.detach(), pb.detach()
        hd, h0d, h1d = h.detach(), h0.detach(), h1.detach()
        v = (p1 - p0 - 0.5 * gv * tf.detach() ** 2) / tf.detach()
        v0 = v.clone()
        E = torch.eye(3, device=pa.device)
        Jm = tf.detach()[..., None] * E

        for _ in range(iters):
            V4 = torch.cat([v] + [v + eps * E[i] for i in range(3)])
            e = _end(p0.repeat(4, 1), V4, hd.repeat(4, 1), h0d.repeat(4, 1), h1d.repeat(4, 1), n.repeat(4))
            e0 = e[:M]
            Jm = torch.stack([(e[(i + 1) * M:(i + 2) * M] - e0) / eps for i in range(3)], -1) + 1e-3 * tf.detach()[..., None] * E
            step = torch.linalg.solve(Jm, (p1 - e0)[..., None])[..., 0]
            v = torch.where(torch.isfinite(step).all(-1, keepdim=True), v + step.clamp(-80, 80), v).clamp(-150, 150)
        bad = ~(torch.isfinite(Jm).all((1, 2)) & torch.isfinite(v).all(-1))
        v = torch.where(bad[:, None], v0, v)
        Jm = torch.where(bad[:, None, None], tf.detach()[..., None] * E, Jm)
    e0 = _end(pa, v, h, h0, h1, n)
    v = v + torch.linalg.solve(Jm, (pb - e0)[..., None])[..., 0].clamp(-80, 80)
    return _sim(pa, v, h, N, h0, None), n, f0.long()


def _end(p, v, h, h0, h1, n):
    """Position at the flight's end time: whole frames up to n, then the partial last frame (per flight)."""
    gv = torch.tensor([0, 0, -G], device=p.device)
    N = int(n.max())

    def step(p, v, hh):
        for _ in range(SUB):
            v = v + hh * (gv - K_DRAG * v.norm(dim=-1, keepdim=True) * v)
            p = p + hh * v
        return p, v
    p, v = step(p, v, h0)
    for k in range(N):
        live = (k < n)[:, None]
        p2, v2 = step(p, v, h)
        p, v = torch.where(live, p2, p), torch.where(live, v2, v)
    p, v = step(p, v, h1)
    return p


def assemble(K, hits, dt, span=None, frac=None):
    """Shuttle path (B,T,3) from the contact points K (B,T,3) at the hit frames and the two ends of the flight
    (span: (B,2) first / last frame the shuttle is in play, default the window's ends; outside it holds still).
    frac (B,T) in [-0.5, 0.5]: when within its frame each hit happens (the knot is the shuttle at that moment)."""
    B, T = K.shape[:2]
    dev = K.device
    pa, pb, ta, tb, dts = [], [], [], [], []
    hits = hits.detach().cpu().numpy() if torch.is_tensor(hits) else hits
    ftime = torch.arange(T, device=dev, dtype=torch.float32)[None].expand(B, T)
    if frac is not None:
        ftime = ftime + frac
    plan = []
    for bi in range(B):
        s0, s1 = (0, T - 1) if span is None else (int(span[bi][0]), int(span[bi][1]))
        ks = sorted(set([s0, s1] + [int(t) for t in np.where(hits[bi])[0] if s0 < t < s1]))
        if len(ks) < 2:
            ks = [0, T - 1]
        for a, c in zip(ks[:-1], ks[1:]):
            plan.append((bi, a, c))
    ia = torch.tensor([(b_, a) for b_, a, _ in plan], device=dev)
    ib = torch.tensor([(b_, c) for b_, _, c in plan], device=dev)
    dtt = dt.reshape(-1)[ia[:, 0]] if torch.is_tensor(dt) else torch.full((len(plan),), float(dt), device=dev)
    tA, tB = ftime[ia[:, 0], ia[:, 1]], ftime[ib[:, 0], ib[:, 1]]
    tB = torch.maximum(tB, tA + 0.5)
    tr, n, f0 = flights(K[ia[:, 0], ia[:, 1]], K[ib[:, 0], ib[:, 1]], tA, tB, dtt)
    # every frame takes the flight it falls in (frames between flights' sample ranges take the nearest sample)
    fid = torch.zeros(B, T, dtype=torch.long, device=dev)
    off = torch.zeros(B, T, dtype=torch.long, device=dev)
    f0c, nc = f0.cpu().numpy(), n.cpu().numpy()
    fid_np, off_np = np.zeros((B, T), int), np.zeros((B, T), int)
    first = {}
    for k, (bi, a, c) in enumerate(plan):
        lo, hi = int(f0c[k]), int(f0c[k] + nc[k])
        first.setdefault(bi, (k, lo))
        for t in range(max(lo, 0), min(hi, T - 1) + 1):
            fid_np[bi, t], off_np[bi, t] = k, t - lo
        last = (k, lo, hi)
        if k + 1 == len(plan) or plan[k + 1][0] != bi:  # after the last flight: hold its end
            fid_np[bi, hi + 1:], off_np[bi, hi + 1:] = k, nc[k]
            k0, lo0 = first[bi]
            fid_np[bi, :max(lo0, 0)], off_np[bi, :max(lo0, 0)] = k0, 0
    return tr[torch.as_tensor(fid_np, device=dev), torch.as_tensor(off_np, device=dev)]

def _phuber(x, c):
    return torch.sqrt(x ** 2 + c ** 2) - c


def losses(out, b, w=None):
    w = w or {}
    dev = b["C"].device
    dt = b["dt"][:, None]
    pres = b["present"]
    J3 = out["joints"]
    # ---- E_obs: detections on their rays
    e = _angle_px(J3, b["C"][:, :, None, None], b["kp_dir"])
    cw = b["kp_conf"] * pres[..., None]
    l_rep = (F.huber_loss(e, torch.zeros_like(e), delta=5.0, reduction="none") * cw).sum() / (cw.sum() + 1e-6)
    S = out["shuttle"]
    sv = b["sh_vis"]
    l_srep = (F.huber_loss(_angle_px(S, b["C"], b["sh_dir"]), torch.zeros_like(sv), delta=3.0, reduction="none") * sv).sum() / (sv.sum() + 1e-6)
    # racket: each hand's racket head against the detected boxes, weighted by how likely that hand holds it
    pi2 = torch.stack([1 - out["pi"], out["pi"]], -1)[:, None]  # (B,1,P,2)
    heads2 = out["w2"] + GRIP_TO_HEAD * out["u2"]  # (B,T,P,2,3)
    ang = _angle_px(heads2[..., None, :], b["C"][:, :, None, None, None], b["rk_dir"][:, :, None, None])  # (B,T,P,2,R)
    tol = b["rk_size"][:, :, None, None] * 1000.0
    valid = (b["rk_conf"][:, :, None, None] > 0.25) & (ang < 2 * tol + 30)
    # a box is that hand's only if it is in reach of the detected wrist: otherwise the arm gets dragged onto another
    # player's racket (the head-only test above lets the model move its head to whichever box it likes)
    kw = b["kp_dir"][..., [9, 10], :]  # (B,T,P,2,3)
    reach = (kw[..., None, :] - b["rk_dir"][:, :, None, None]).norm(dim=-1) * 1000.0  # (B,T,P,2,R)
    seen = (b["kp_conf"][..., [9, 10]] > 0.3)[..., None]
    valid = valid & (~seen | (reach < tol + 20))
    best = torch.where(valid, ang, torch.full_like(ang, 1e4)).min(-1).values  # (B,T,P,2)
    has = (best < 1e4).float() * pres[..., None]
    lr = F.huber_loss(best.clamp(max=500), torch.zeros_like(best), delta=10.0, reduction="none")
    lr = torch.where(has > 0, lr, torch.full_like(lr, 60.0))  # a hand with no box near it scores as a miss
    anyb = (has.sum(-1) > 0).float()
    l_rk = ((lr * pi2).sum(-1) * anyb).sum() / (anyb.sum() + 1e-6) * 0.1
    # ---- E_force
    # shuttle: flights are exact between contacts; a contact happens at the hitter's racket head (robust: the racket
    # head is only roughly known), and the shuttle stays above the floor
    ok = b["cam_ok"]
    hk = out["hits"] * ok
    con = (out["knot"] - out["racket"]).norm(dim=-1)
    l_con = (F.huber_loss(con, torch.zeros_like(con), delta=0.3, reduction="none") * hk).sum() / (hk.sum() + 1.0)
    l_sfloor = F.relu(-S[..., 2]).mean()
    # players: the pelvis is held up only while a foot is on the floor (else it falls)
    pel = out["pel"]
    acc = (pel[:, 2:] - 2 * pel[:, 1:-1] + pel[:, :-2]) / dt[:, :, None, None] ** 2
    f = acc - torch.tensor([0, 0, -G], device=dev)  # force per unit mass the body must get from somewhere
    zf = torch.minimum(J3[..., 15, 2], J3[..., 16, 2])[:, 1:-1]
    air = torch.sigmoid((zf - ANKLE_H - 0.08) / 0.03)
    p3 = pres[:, 2:] * pres[:, 1:-1] * pres[:, :-2]
    l_sup = ((_phuber(f.norm(dim=-1), 2.0) * air + 0.02 * _phuber(f.norm(dim=-1), 20.0)) * p3).sum() / (p3.sum() + 1e-6)
    # the floor is solid
    z = J3[..., 2]
    l_floor = ((F.relu(ANKLE_H - z[..., 15:17]).sum(-1) + F.relu(-z).sum(-1)) * pres).sum() / (pres.sum() + 1e-6)
    # effort: joints accelerate little (robust: swings are allowed)
    ja = (J3[:, 2:] - 2 * J3[:, 1:-1] + J3[:, :-2]).norm(dim=-1) / dt[:, :, None, None] ** 2
    l_eff = (_phuber(ja, 5.0) * p3[..., None]).sum() / (p3.sum() * J + 1e-6)
    u = out["u2"]
    ua = (u[:, 2:] - 2 * u[:, 1:-1] + u[:, :-2]).norm(dim=-1) / dt[:, :, None, None] ** 2  # rad/s^2
    l_rks = ((_phuber(ua, 20.0) * pi2).sum(-1) * p3).sum() / (p3.sum() + 1e-6)
    # ---- E_prior: learned pose prior; triangulated joints and hit labels where a dataset has them
    l_prior = torch.zeros((), device=dev)
    pp = RN.prior(dev)
    if pp is not None:
        on = pres > 0
        l_prior = pp.penalty(J3[on]).mean() if on.any() else l_prior
    l_j3d = torch.zeros((), device=dev)
    if "J3d_mask" in b:
        jm = b["J3d_mask"]
        l_j3d = (F.huber_loss(J3, b["J3d"], delta=0.1, reduction="none").sum(-1) * jm).sum() / (jm.sum() + 1e-6)
    # VIMO teacher (training only): bone directions where VIMO agrees with the keypoints -- depth of each bone
    l_vt = torch.zeros((), device=dev)
    if "Vd_mask" in b:
        P_ = lambda i: J3[..., i, :]
        ends = [(P_(12), P_(11)), (P_(11), P_(13)), (P_(13), P_(15)), (P_(12), P_(14)), (P_(14), P_(16)),
                (0.5 * (P_(11) + P_(12)), 0.5 * (P_(5) + P_(6))), (P_(6), P_(5)), (P_(5), P_(7)), (P_(7), P_(9)),
                (P_(6), P_(8)), (P_(8), P_(10))]
        Dm = torch.stack([_unit(c - a) for a, c in ends], -2)  # (B,T,P,11,3)
        vm = b["Vd_mask"] * pres[..., None]
        l_vt = ((1 - (Dm * b["Vd"]).sum(-1)) * vm).sum() / (vm.sum() + 1e-6)
    l_hit = torch.zeros((), device=dev)
    if "hit_lbl" in b:  # which player hits when (BFMD train matches, FineBadminton-20K)
        lbl = F.max_pool1d(b["hit_lbl"].permute(0, 2, 1), 3, stride=1, padding=1).permute(0, 2, 1)
        hm = b["hit_has"][:, None, None] * pres
        bce = F.binary_cross_entropy_with_logits(out["hit"], lbl, pos_weight=torch.tensor(10.0, device=dev), reduction="none")
        l_hit = (bce * hm).sum() / (hm.sum() + 1e-6)
    if "hit_any" in b:  # when, not who (DroneData dev matches)
        la = F.max_pool1d(b["hit_any"][:, None], 3, stride=1, padding=1)[:, 0]
        ha = b["hit_any_has"][:, None] * ok
        pa = out["gate"].clamp(1e-5, 1 - 1e-5)
        l_hit = l_hit + (-(10.0 * la * torch.log(pa) + (1 - la) * torch.log(1 - pa)) * ha).sum() / (ha.sum() + 1e-6)
    parts = {"rep": l_rep, "srep": l_srep, "rk": l_rk,
             "con": l_con * 5.0, "sfloor": l_sfloor * 10.0, "sup": l_sup * 0.5, "floor": l_floor * 50, "eff": l_eff * 0.2, "rks": l_rks * 0.01,
             "prior": l_prior * 20, "j3d": l_j3d * 20, "hit": l_hit * 5, "vt": l_vt * VTEACH}
    if not SHUTTLE_TRAIN:
        for k in ("srep", "con", "sfloor"):
            parts[k] = parts[k] * 0.0
    parts = {k: v * w.get(k, 1.0) for k, v in parts.items()}
    return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}




MIN_GAP = 8  # frames: two hits closer than this are one (keep the more likely)


def rally_hits(gate):
    """Hit frames from the network's (rally-averaged) hit probability: peaks over HIT_THR, at least MIN_GAP apart."""
    from scipy.signal import find_peaks
    return find_peaks(gate, height=HIT_THR, distance=MIN_GAP)[0]


CHAIN_MAXDEV = float(os.environ.get("CHAIN_MAXDEV", "1.5"))


def chain_refit(S_init, hits, f, iters=60, shots=None):
    """The shuttle as one continuous chain of gravity + drag flights between contacts (exact: no jumps, smooth by
    construction), fitted to the shuttle's rays. Contacts start where S_init has the shuttle at the hits (the
    hitter-anchored per-shot fit) and stay near it; each contact also gets its moment within its frame. numpy.
    shots: the per-shot fit's own (start, end) frames -- its shot starts (which include the splits it made where one
    flight could not explain the track, e.g. a missed hit) become the contacts, and its last end the chain's end."""
    T = len(S_init)
    vis_np = f["sh_vis"] > 0
    if vis_np.sum() < 5:
        return S_init
    vv = np.where(vis_np)[0]
    span = (int(vv[0]), int(vv[-1]))
    t0 = np.zeros(T, np.float32)  # sub-frame contact times
    if shots:
        span = (max(0, int(round(shots[0][0]))), min(T - 1, int(round(shots[-1][1]))))
        hits = []
        for a, _ in shots[1:]:
            k = int(round(a))
            hits.append(k)
            if 0 <= k < T:
                t0[k] = np.clip(a - k, -0.45, 0.45)
    H = np.zeros((1, T), bool)
    H[0, [h for h in hits if span[0] < h < span[1]]] = True
    knots = [span[0], span[1]] + [int(h) for h in np.where(H[0])[0]]
    K0 = torch.tensor(np.nan_to_num(S_init), dtype=torch.float32)
    Kp = K0.clone().requires_grad_(True)
    ph = torch.tensor(np.arctanh(2 * t0), requires_grad=True)
    C = torch.tensor(f["C"], dtype=torch.float32)
    d = torch.tensor(f["sh_dir"], dtype=torch.float32)
    vis = torch.tensor(vis_np)
    ki = torch.tensor(sorted(set(knots)))
    sp = torch.zeros(T, dtype=torch.bool)
    sp[span[0]:span[1] + 1] = True
    opt = torch.optim.LBFGS([Kp, ph], lr=1, max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        S = assemble(Kp[None], H, float(f["dt"]), span=[span], frac=(0.5 * torch.tanh(ph))[None])[0]
        e = _angle_px(S, C, d)[vis]
        loss = F.huber_loss(e, torch.zeros_like(e), delta=3.0) + 10.0 * F.relu(-S[:, 2]).mean()             + 5.0 * F.huber_loss((Kp[ki] - K0[ki]).norm(dim=-1), torch.zeros(len(ki)), delta=0.3)             + 20.0 * F.huber_loss((S - K0)[sp].norm(dim=-1), torch.zeros(int(sp.sum())), delta=0.3)  # stay near the fit
        opt.zero_grad(); loss.backward()
        return loss
    try:
        opt.step(closure)
    except Exception:  # a degenerate solve: keep the per-shot fit
        return S_init
    with torch.no_grad():
        S = assemble(Kp[None], H, float(f["dt"]), span=[span], frac=(0.5 * torch.tanh(ph))[None])[0].numpy()
    dev_ = np.linalg.norm(S - np.nan_to_num(S_init), axis=1)[span[0]:span[1] + 1]
    if os.environ.get("CHAIN_DEBUG"):
        print("chain_refit: finite", bool(np.isfinite(S).all()), "max dev %.2f at %d" % (np.nanmax(dev_), span[0] + int(np.nanargmax(dev_))), flush=True)
    if not np.isfinite(S).all() or dev_.max() > CHAIN_MAXDEV:  # the chain could not follow the fit: keep the fit
        return S_init
    S[:, 2] = np.maximum(S[:, 2], 0)
    return S
