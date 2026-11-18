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
VTEACH = float(os.environ.get("VTEACH", "0"))
S3D_W = float(os.environ.get("SHUTTLE3D", "0"))  # two-view 3D shuttle supervision weight (DroneData training matches)  # VIMO bone-direction teacher weight (tools/train_rallynet.vimo_teacher)
CON3D_W = float(os.environ.get("CONTACT3D", "0"))  # physical contact points: the hitter's racket head is there at a hit
XVIEW_W = float(os.environ.get("XVIEW", "0"))  # second fixed camera (CCTV side view): the 3D shuttle on its rays
TCON_W = float(os.environ.get("TCON", "0"))  # the teacher's hits: the student's racket head meets the teacher's shuttle
CCTV3D_W = float(os.environ.get("CCTV3D", "0"))  # second fixed camera (CCTV side view): contacts at the two-view shuttle
RTOP_W = float(os.environ.get("RACKET_TOP", "0"))  # second-view rackets: the racket's middle is on a top-drone racket ray
RK_DETACH = os.environ.get("RK_DETACH", "0") in ("1", "obs", "body")  # racket observations move the racket, not the arm (the pose terms own it)
C3D_DETACH = os.environ.get("RK_DETACH", "0") == "1"  # ... also at contacts ("obs": a hit may still pull the hand to the shuttle)
RK_BODY = os.environ.get("RK_DETACH", "0") == "body"  # "body": ... but they may still move the whole player (its root)
STUDENT = os.environ.get("STUDENT", "0") == "1"  # impulse-sparse learned dynamics: the shuttle is the network's output
# m: the student's highest shuttle when it is off the TV ray (0 = the original box, z in [-2.5, 5.5]); high lifts reach 6-7 m
S_ZMAX = float(os.environ.get("S_ZMAX", "0"))
TEACH_W = float(os.environ.get("TEACH", "0"))  # weight of the two-view teacher's dense 3D (tools/cctv_teach.py)
TMOT_W = float(os.environ.get("TMOT", "0"))  # teacher shuttle velocity + acceleration matching (student)
TSMOOTH_W = float(os.environ.get("TSMOOTH", "0"))  # control: penalise the shuttle's deviation from free fall (no teacher motion)
DYN_W = float(os.environ.get("DYN", "1"))  # weight of the dynamics energy (student)
# TRAM-format bodies from the same network (24 SMPL joints), trained on the multi-camera teacher's TRAM bodies
BODY24_W = float(os.environ.get("BODY24", "0"))
# student shuttle as a free 3D position every frame (the TV ray is a loss, not a constraint): detector pixel noise is
# not copied into 3D, so the network can follow a smooth flight through noisy detections
FREE_S = os.environ.get("FREE_S", "0") == "1"
# SMPL joints built on the network's own skeleton: index of the COCO joint each one starts from (-1: built below)
FOOT_W = float(os.environ.get("FOOT", "0"))  # no-foot-skating weight on the TRAM body's ankles and feet (m/s)
FOOTV_W = float(os.environ.get("FOOTV", "0"))  # teacher-velocity matching of the TRAM body's pelvis, ankles and feet
B24_SHAPE = float(os.environ.get("B24_SHAPE", "0"))  # extra weight (x BODY24) on the TRAM body's shape and limb angles
_SMPL_PAR = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]  # SMPL parent of joint 1..23
_SMPL_KID = list(range(1, 24))
_SMPL_FROM = {1: 11, 2: 12, 4: 13, 5: 14, 7: 15, 8: 16, 16: 5, 17: 6, 18: 7, 19: 8, 20: 9, 21: 10}
AERO = os.environ.get("AERO", "learn")  # "learn": drag coefficient + a learned force; "fixed": quadratic drag at V_TERM
RK_MID = 0.38  # m, hand to the racket box centre (median of the two-drone triangulated rackets, tools/triangulate_rackets.py)


def _unit(v, eps=1e-6):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


class Aero(nn.Module):
    """The shuttle's aerodynamic acceleration a(v), one law shared by every rally: drag along -v with a learned
    coefficient plus learned corrections along -v and up, functions of the speed and the climb angle only (no
    preferred horizontal direction). Starts as quadratic drag with terminal velocity V_TERM."""

    def __init__(self):
        super().__init__()
        self.logk = nn.Parameter(torch.tensor(math.log(K_DRAG)), requires_grad=AERO == "learn")
        self.mlp = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, v):
        s = v.norm(dim=-1, keepdim=True)
        u = v / (s + 1e-6)
        a = -torch.exp(self.logk) * s * v
        if AERO == "learn":
            c = self.mlp(torch.cat([s / 30.0, u[..., 2:3]], -1)) * 5.0  # m/s^2
            a = a - c[..., :1] * u + c[..., 1:2] * torch.tensor([0, 0, 1.0], device=v.device)
        return a


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
        if STUDENT:  # per frame: depth along the ray (1), position when unseen (3), velocity (3)
            self.f_out = nn.Linear(d, 7)
            nn.init.zeros_(self.f_out.weight); nn.init.zeros_(self.f_out.bias)
            self.aero = Aero()
        if BODY24_W > 0:  # per player and frame: offset of each of the 24 SMPL joints from its skeleton anchor
            self.b_out = nn.Linear(d, 24 * 3)
            nn.init.zeros_(self.b_out.weight); nn.init.zeros_(self.b_out.bias)

    def smpl24(self, xp, joints, pel):
        """TRAM-format body (B,T,P,24,3): each SMPL joint starts at the matching joint of the network's own skeleton
        (spine and neck along pelvis -> neck, collars between neck and shoulders, hands past the wrists), plus a
        learned offset of at most 0.3 m. Limbs, hands and the racket share one skeleton."""
        B, T = joints.shape[:2]
        jt = lambda i: joints[..., i, :]
        neck = 0.5 * (jt(5) + jt(6))
        up = _unit(neck - pel)
        base = [None] * 24
        for s, c in _SMPL_FROM.items():
            base[s] = jt(c)
        base[0] = pel
        for s, a in ((3, 0.25), (6, 0.5), (9, 0.75)):
            base[s] = pel + a * (neck - pel)
        base[10], base[11] = jt(15), jt(16)
        base[12] = neck + 0.08 * up
        base[13], base[14] = 0.5 * (neck + jt(5)), 0.5 * (neck + jt(6))
        base[15] = jt(0)
        base[22] = jt(9) + 0.08 * _unit(jt(9) - jt(7))
        base[23] = jt(10) + 0.08 * _unit(jt(10) - jt(8))
        return torch.stack(base, -2) + 0.3 * torch.tanh(self.b_out(xp).reshape(B, T, P, 24, 3))

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
        us, ws, urs = [], [], []
        for k, (el, wr, sh) in enumerate([(lel, lwr, lsh), (rel, rwr, rsh)]):
            a = _unit(wr - el)
            ref = _unit(sh - el) + 0.3 * D[..., 6, :]
            n = _unit(torch.cross(a, ref, dim=-1))
            c = torch.cross(n, a, dim=-1)
            ul = po[..., -7 + 3 * k:po.shape[-1] - 4 + 3 * k] + torch.tensor([1.0, 0, 0], device=dev)
            us.append(_unit(ul[..., :1] * a + ul[..., 1:2] * n + ul[..., 2:3] * c))
            ws.append(wr)
            if RK_DETACH:  # the same racket in a frozen forearm frame: what the racket terms train
                urs.append(_unit(ul[..., :1] * a.detach() + ul[..., 1:2] * n.detach() + ul[..., 2:3] * c.detach()))
        u2, w2 = torch.stack(us, -2), torch.stack(ws, -2)  # (B,T,P,2,3)
        self._u2r = torch.stack(urs, -2) if RK_DETACH else u2
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
        extra = {}
        if STUDENT:  # the shuttle is the network's own per-frame state; its dynamics are scored in losses()
            fo = self.f_out(x[:, :, P])
            ctr = torch.as_tensor(CENTRE, device=b["C"].device) + torch.tensor([0, 0, 1.5], device=b["C"].device)
            lam = (ctr - b["C"]).norm(dim=-1, keepdim=True) * torch.exp(torch.tanh(fo[..., :1]))
            half = torch.tensor([4.0, 9.0, 4.0], device=b["C"].device)
            if S_ZMAX > 0:  # same lowest point (-2.5 m), higher ceiling
                ctr = ctr + torch.tensor([0, 0, (S_ZMAX - 5.5) / 2], device=b["C"].device)
                half = torch.tensor([4.0, 9.0, (S_ZMAX + 2.5) / 2], device=b["C"].device)
            if FREE_S:
                S = ctr + half * torch.tanh(fo[..., 1:4])
            else:
                S = torch.where((b["sh_vis"] > 0)[..., None], b["C"] + lam * b["sh_dir"], ctr + half * torch.tanh(fo[..., 1:4]))
            Vel = 10.0 * fo[..., 4:7]
            gv = torch.tensor([0, 0, -G], device=S.device)
            dtt = b["dt"][:, None, None]
            extra = {"svel": Vel, "dv": (Vel[:, 1:] - Vel[:, :-1]) - (gv + self.aero(Vel[:, :-1])) * dtt,  # unexplained velocity change (m/s)
                     "dx": (S[:, 1:] - S[:, :-1]) - 0.5 * (Vel[:, 1:] + Vel[:, :-1]) * dtt}  # position vs velocity (m)
        if BODY24_W > 0:
            extra["joints24"] = self.smpl24(x[:, :, :P], joints, pel)
        return {**extra, "joints": joints, "racket_u": u, "hit": hl, "shuttle": S, "knot": K, "racket": racket,
                "hits": hits.float(), "gate": gate, "len": Lp, "pel": pel, "head": head, "u2": u2, "w2": w2, "u2r": self._u2r, "pi": pi,
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
    rw2 = out["w2"].detach() if RK_DETACH else out["w2"]  # the hand the racket terms see
    if RK_BODY:
        pel_ = out["pel"][..., None, :]
        rw2 = pel_ + (out["w2"] - pel_).detach()
    ru2 = out.get("u2r", out["u2"])
    rheads2 = rw2 + GRIP_TO_HEAD * ru2
    ang = _angle_px(rheads2[..., None, :], b["C"][:, :, None, None, None], b["rk_dir"][:, :, None, None])  # (B,T,P,2,R)
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
    # two-view 3D shuttle where a training rally has it (DroneData training matches): the flights chained through the
    # network's contact points must follow it -- this is what teaches the contacts their depth
    l_s3d = torch.zeros((), device=dev)
    if S3D_W > 0 and "S3d_mask" in b:
        rows = torch.where(b["S3d_mask"].sum(1) > 10)[0]
        if len(rows):
            Sa = out["shuttle"][rows] if STUDENT else assemble(out["knot"][rows], out["hits"][rows] > 0, b["dt"][rows])
            m3 = b["S3d_mask"][rows]
            l_s3d = (F.huber_loss(Sa, b["S3d"][rows], delta=0.3, reduction="none").sum(-1) * m3).sum() / (m3.sum() + 1e-6)
    # physical contacts (DroneData training matches): at a labelled hit the nearest player's racket head (racket hand
    # by the network's hand belief) is at the contact point recovered from the two-view shuttle's flight after it
    l_c3d = torch.zeros((), device=dev)
    if CON3D_W > 0 and "Con3d_mask" in b and b["Con3d_mask"].sum() > 0:
        hd = ((rheads2 if C3D_DETACH else heads2) * pi2[..., None]).sum(-2)  # (B,T,P,3)
        dd = (hd - b["Con3d"][:, :, None]).norm(dim=-1)
        dd = torch.where(pres > 0, dd, torch.full_like(dd, 1e3)).min(-1).values  # (B,T)
        cm = b["Con3d_mask"]
        l_c3d = (F.huber_loss(dd.clamp(max=5.0), torch.zeros_like(dd), delta=0.3, reduction="none") * cm).sum() / (cm.sum() + 1e-6)
    # second fixed camera (CCTV rallies, no hit labels): wherever the network itself puts a hit, the nearest player's
    # racket head must be where the TV and side cameras together saw the shuttle -- the same contact physics as above,
    # with the network's own timing
    l_c3c = torch.zeros((), device=dev)
    if CCTV3D_W > 0 and "S3c_mask" in b and b["S3c_mask"].sum() > 0:
        hd = ((rheads2 if C3D_DETACH else heads2) * pi2[..., None]).sum(-2)  # (B,T,P,3)
        dd = (hd - b["S3c"][:, :, None]).norm(dim=-1)
        dd = torch.where(pres > 0, dd, torch.full_like(dd, 1e3)).min(-1).values  # (B,T)
        cm = b["S3c_mask"] * out["hits"]
        l_c3c = (F.huber_loss(dd.clamp(max=5.0), torch.zeros_like(dd), delta=0.3, reduction="none") * cm).sum() / (cm.sum() + 1e-6)
    # teacher contacts (CCTV training rallies): at each of the multi-camera teacher's hits the hitter's racket head (by the
    # network's hand belief; the hit may pull the arm) is at the teacher's shuttle, within +-1 frame of the teacher's timing
    l_tcon = torch.zeros((), device=dev)
    if TCON_W > 0 and "Tch_mask" in b and b["Tch_mask"].sum() > 0:
        hd = (heads2 * pi2[..., None]).sum(-2)  # (B,T,P,3)
        dd = (hd - b["Tch"][:, :, None]).norm(dim=-1)
        dd = torch.where(pres > 0, dd, torch.full_like(dd, 1e3)).min(-1).values.clamp(max=5.0)  # (B,T)
        dd = -F.max_pool1d(-dd[:, None], 3, stride=1, padding=1)[:, 0]
        cm = b["hit_any"] * b["Tch_mask"] * (b["hit_any_m"] if "hit_any_m" in b else 1.0)
        l_tcon = (F.huber_loss(dd, torch.zeros_like(dd), delta=0.3, reduction="none") * cm).sum() / (cm.sum() + 1e-6)
    # second fixed camera (CCTV rallies, unlabelled): the shuttle the network builds from the TV view alone must also lie
    # on the side camera's rays through its own detections -- the other view is the label, every frame it saw the shuttle
    l_xv = torch.zeros((), device=dev)
    if XVIEW_W > 0 and "Xs_vis" in b and b["Xs_vis"].sum() > 0:
        xv = b["Xs_vis"]
        l_xv = (F.huber_loss(_angle_px(S, b["Xs_C"], b["Xs_dir"]), torch.zeros_like(xv), delta=3.0, reduction="none") * xv).sum() / (xv.sum() + 1e-6)
    # second view (DroneData training matches): the top drone's racket boxes. The middle of each hand's racket must lie
    # on a top-view ray through a box centre that passes within reach of that hand -- the back view alone cannot tell
    # a racket held up from one pointing away, the top view can
    l_rtop = torch.zeros((), device=dev)
    if RTOP_W > 0 and "Tk_m" in b and b["Tk_m"].sum() > 0:
        Ct, Dt, tm = b["Tk_C"][:, :, None, None, None], b["Tk_dir"][:, :, None, None], b["Tk_m"][:, :, None, None] > 0

        def rdist(X):  # (B,T,P,2,3) -> distance to each top ray (B,T,P,2,RT)
            v = X[..., None, :] - Ct
            return (v - (v * Dt).sum(-1, keepdim=True) * Dt).norm(dim=-1)
        near = tm & (rdist(rw2) < 0.45)
        dm = torch.where(near, rdist(rw2 + RK_MID * ru2), torch.full_like(near, 1e3, dtype=heads2.dtype)).min(-1).values
        has = near.any(-1) & (pres[..., None] > 0)  # (B,T,P,2)
        wt = pi2.expand_as(dm) * has
        l_rtop = (F.huber_loss(dm.clamp(max=2.0), torch.zeros_like(dm), delta=0.1, reduction="none") * wt).sum() / (wt.sum() + 1e-6)
    # student: impulse-sparse dynamics -- between hits the shuttle's velocity changes only by gravity + the learned
    # aerodynamic force; an unexplained change is free only where the network itself says a hit happens, and hits
    # are rare (on rallies without hit labels the gate pays for itself)
    l_dyn = l_kin = l_sp = l_tch = l_fl = l_tmot = torch.zeros((), device=dev)
    if STUDENT:
        okp = b["cam_ok"][:, 1:] * b["cam_ok"][:, :-1]
        gp = torch.maximum(out["gate"][:, 1:], out["gate"][:, :-1])
        l_dyn = (_phuber(out["dv"].norm(dim=-1), 0.3) * (1 - gp) * okp).sum() / (okp.sum() + 1e-6)
        l_kin = (_phuber(out["dx"].norm(dim=-1), 0.02) * (1 - 0.9 * gp) * okp).sum() / (okp.sum() + 1e-6)
        unl = (1 - torch.maximum(b["hit_has"], b["hit_any_has"]))[:, None] * b["cam_ok"]
        l_sp = (out["gate"] * unl).sum() / (unl.sum() + 1e-6)
        l_fl = (F.relu(-out["shuttle"][..., 2]) * b["cam_ok"]).sum() / (b["cam_ok"].sum() + 1e-6)
        if TEACH_W > 0 and "Tch_mask" in b:
            tm = b["Tch_mask"]
            l_tch = (F.huber_loss(out["shuttle"], b["Tch"], delta=0.3, reduction="none").sum(-1) * tm).sum() / (tm.sum() + 1e-6)
            if TMOT_W > 0:
                Ss, St = out["shuttle"], b["Tch"]
                v, vt, m1 = Ss[:, 1:] - Ss[:, :-1], St[:, 1:] - St[:, :-1], tm[:, 1:] * tm[:, :-1]
                a, at, m2 = v[:, 1:] - v[:, :-1], vt[:, 1:] - vt[:, :-1], m1[:, 1:] * m1[:, :-1]
                l_tmot = ((F.huber_loss(v, vt, delta=0.05, reduction="none").sum(-1) * m1).sum() / (m1.sum() + 1e-6)
                          + (F.huber_loss(a, at, delta=0.05, reduction="none").sum(-1) * m2).sum() / (m2.sum() + 1e-6))
    l_tsm = torch.zeros((), device=dev)
    if TSMOOTH_W > 0:  # the penalty counterpart of l_tmot: acceleration away from gravity, every frame the camera saw
        Ss = out["shuttle"]
        a = Ss[:, 2:] - 2 * Ss[:, 1:-1] + Ss[:, :-2]
        g = torch.tensor([0.0, 0.0, -9.81], device=dev) * b["dt"].reshape(-1, 1, 1) ** 2
        m2 = b["cam_ok"][:, 2:] * b["cam_ok"][:, 1:-1] * b["cam_ok"][:, :-2]
        l_tsm = (F.huber_loss(a - g, torch.zeros_like(a), delta=0.05, reduction="none").sum(-1) * m2).sum() / (m2.sum() + 1e-6)
    # TRAM-format bodies: the teacher's multi-camera TRAM bodies (CCTV training rallies, frames >= 2 cameras saw the
    # player), plus the same low-effort motion as the skeleton (no frame-to-frame jitter in the offsets)
    l_b24 = torch.zeros((), device=dev)
    if BODY24_W > 0 and "joints24" in out:
        J24 = out["joints24"]
        if "B24_mask" in b:
            bm = b["B24_mask"] * pres
            l_b24 = (F.huber_loss(J24, b["B24"], delta=0.1, reduction="none").sum(-1).mean(-1) * bm).sum() / (bm.sum() + 1e-6)
            if B24_SHAPE > 0:  # the body's own shape, apart from where it stands: pelvis-relative joints, bone
                # directions (the limb angles) and bone lengths of the teacher's TRAM body, at full strength
                rel, rt = J24 - J24[..., :1, :], b["B24"] - b["B24"][..., :1, :]
                l_rel = ((rel - rt).pow(2).sum(-1) + 1e-6).sqrt().mean(-1)  # pelvis-relative MPJPE (m)
                bs, bt = J24[..., _SMPL_KID, :] - J24[..., _SMPL_PAR, :], b["B24"][..., _SMPL_KID, :] - b["B24"][..., _SMPL_PAR, :]
                ls, lt = bs.norm(dim=-1), bt.norm(dim=-1)
                l_dir = (1 - (bs * bt).sum(-1) / (ls * lt + 1e-6)).mean(-1)  # limb angles
                l_len = (ls - lt).abs().mean(-1)  # bone lengths (m)
                l_b24 = l_b24 + B24_SHAPE * ((l_rel + l_dir + l_len) * bm).sum() / (bm.sum() + 1e-6)
        if FOOT_W > 0:  # no foot skating: an ankle or foot on the floor (both frames) does not slide -- the body has
            # to step. Contact is soft, from the joint's own height (detached: the penalty never lifts a foot by itself)
            fj = J24[..., [7, 8, 10, 11], :]
            hc = torch.tensor([0.11, 0.11, 0.06, 0.06], device=dev)
            c = torch.sigmoid((hc - fj[..., 2]) / 0.015).detach()
            cc = c[:, 1:] * c[:, :-1] * (pres[:, 1:] * pres[:, :-1])[..., None]
            vxy = ((fj[:, 1:, ..., :2] - fj[:, :-1, ..., :2]).pow(2).sum(-1) + 1e-8).sqrt() / dt[:, :, None, None]
            l_b24 = l_b24 + FOOT_W / BODY24_W * (vxy * cc).sum() / (cc.sum() + 1e-6)
        if FOOTV_W > 0 and "B24_mask" in b:  # step like the teacher: pelvis, ankle and foot velocities (m/s) follow the
            # teacher's TRAM body, so a planted teacher foot stays planted while the body still travels at its speed
            jv = [0, 7, 8, 10, 11]
            vs = (J24[:, 1:, :, jv] - J24[:, :-1, :, jv]) / dt[:, :, None, None, None]
            vt = (b["B24"][:, 1:, :, jv] - b["B24"][:, :-1, :, jv]) / dt[:, :, None, None, None]
            mv = b["B24_mask"][:, 1:] * b["B24_mask"][:, :-1] * pres[:, 1:] * pres[:, :-1]
            lv = F.huber_loss(vs, vt, delta=0.5, reduction="none").sum(-1).mean(-1)
            l_b24 = l_b24 + FOOTV_W / BODY24_W * (lv * mv).sum() / (mv.sum() + 1e-6)
        ja24 = (J24[:, 2:] - 2 * J24[:, 1:-1] + J24[:, :-2]).norm(dim=-1) / dt[:, :, None, None] ** 2
        l_b24 = l_b24 + 0.01 * (_phuber(ja24, 5.0) * p3[..., None]).sum() / (p3.sum() * 24 + 1e-6)
    l_hit = torch.zeros((), device=dev)
    if "hit_lbl" in b:  # which player hits when (BFMD train matches, FineBadminton-20K)
        lbl = F.max_pool1d(b["hit_lbl"].permute(0, 2, 1), 3, stride=1, padding=1).permute(0, 2, 1)
        hm = b["hit_has"][:, None, None] * pres
        bce = F.binary_cross_entropy_with_logits(out["hit"], lbl, pos_weight=torch.tensor(10.0, device=dev), reduction="none")
        l_hit = (bce * hm).sum() / (hm.sum() + 1e-6)
    if "hit_any" in b:  # when, not who (DroneData dev matches)
        la = F.max_pool1d(b["hit_any"][:, None], 3, stride=1, padding=1)[:, 0]
        ha = b["hit_any_has"][:, None] * ok
        if "hit_any_m" in b:
            ha = ha * b["hit_any_m"]
        pa = out["gate"].clamp(1e-5, 1 - 1e-5)
        l_hit = l_hit + (-(10.0 * la * torch.log(pa) + (1 - la) * torch.log(1 - pa)) * ha).sum() / (ha.sum() + 1e-6)
    parts = {"rep": l_rep, "srep": l_srep, "rk": l_rk,
             "con": l_con * 5.0, "sfloor": l_sfloor * 10.0, "sup": l_sup * 0.5, "floor": l_floor * 50, "eff": l_eff * 0.2, "rks": l_rks * 0.01,
             "prior": l_prior * 20, "j3d": l_j3d * 20, "hit": l_hit * 5, "vt": l_vt * VTEACH, "s3d": l_s3d * S3D_W,
             "c3d": l_c3d * CON3D_W, "rtop": l_rtop * RTOP_W, "c3c": l_c3c * CCTV3D_W, "tcon": l_tcon * TCON_W,"xv": l_xv * XVIEW_W,
             "dyn": l_dyn * DYN_W, "kin": l_kin * DYN_W * 10, "sp": l_sp * DYN_W, "tch": l_tch * TEACH_W, "tmot": l_tmot * TMOT_W, "tsm": l_tsm * TSMOOTH_W, "sfl": l_fl * 50,
             "b24": l_b24 * BODY24_W}
    if not SHUTTLE_TRAIN:
        for k in ("srep", "con", "sfloor") if not STUDENT else ("con", "sfloor"):
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
