"""Monocular (back view only) 3D shuttle reconstruction for a whole rally.

Model: a rally is a chain of shots. Shot i starts at hit time tau_i with state (p_i, v_i) and
flies under gravity + quadratic drag until tau_{i+1}. All shots are fit JOINTLY:

  reprojection   robust (soft-L1) pixel residual of every 2D detection          sigma_px
  continuity     shot i must arrive where shot i+1 starts (the shuttle does not
                 teleport at a hit)                                              sigma_cont
  hit anchor     shot start near the hitter's hand, lifted from 2D pose at the
                 feet depth (racket reach makes this loose)                      sigma_hand
  net            shots that cross the net clear the tape, and every hit is on its
                 hitter's side, NET_GAP from the net (hinges)                    sigma_net
  floor          the shuttle never goes below the floor (hinge)                  sigma_floor
  landing        (land=True) when the track ends with the shuttle at rest on the floor, the last
                 shot ends above the rest point (its still pixel ray-cast to the floor; the final
                 fall is near vertical) -- the only depth anchor a rally's last shot has  sigma_land

Birdseye v1 instead fit each shot independently with two hard-ish hand anchors and plain
least squares. The joint chain is what lets well-observed shots fix the depth of poorly
observed neighbours, and the robust loss keeps TrackNet false positives from dragging a shot.
"""
import os

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from .court import L, NET_Y, POST_H

G = np.array([0.0, 0.0, -9.81])
V_TERM = 6.4  # m/s; measured from 3D two-view GT (tools/calibrate_drag.py: median 6.38, IQR 6.22-6.53)
FPS = 30.0
N_GRID = 24  # trajectory samples per shot for the floor and net constraints
NET_GAP = 0.3  # m, a hit is at least this far from the net plane on the hitter's side


@njit(cache=True)
def _rk4(p0, v0, ts_sorted, k, dt):
    out = np.empty((ts_sorted.shape[0], 3))
    x = p0.copy()
    v = v0.copy()
    t = 0.0
    g = np.array([0.0, 0.0, -9.81])
    for i in range(ts_sorted.shape[0]):
        tq = ts_sorted[i]
        while t + dt <= tq + 1e-12:
            s1 = np.sqrt(v @ v); a1 = g - k * s1 * v
            v2 = v + 0.5 * dt * a1; s2 = np.sqrt(v2 @ v2); a2 = g - k * s2 * v2
            v3 = v + 0.5 * dt * a2; s3 = np.sqrt(v3 @ v3); a3 = g - k * s3 * v3
            v4 = v + dt * a3; s4 = np.sqrt(v4 @ v4); a4 = g - k * s4 * v4
            x = x + dt / 6.0 * (v + 2.0 * v2 + 2.0 * v3 + v4)
            v = v + dt / 6.0 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            t += dt
        h = tq - t  # sub-step remainder
        sp = np.sqrt(v @ v)
        out[i] = x + v * h + 0.5 * (g - k * sp * v) * h * h
    return out


def simulate(p0, v0, ts, k=9.81 / V_TERM ** 2, dt=1 / 240.0):
    """Positions at times ts (s, >=0, any order) from state (p0,v0) at t=0. Fixed-step RK4
    (numba; matches scipy's adaptive solver to ~1 mm)."""
    ts = np.asarray(ts, float)
    order = np.argsort(ts)
    out_sorted = _rk4(np.asarray(p0, float), np.asarray(v0, float), ts[order], float(k), float(dt))
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def near_player_px(kps, uv):
    """Pixel distance from uv to the nearest player's pose box widened for racket reach
    (+50% each side, +60% height above the head)."""
    best = 1e9
    for kp in kps:
        k = kp[kp[:, 2] > 0.3]
        if len(k) < 3:
            continue
        x0, y0 = k[:, :2].min(0)
        x1, y1 = k[:, :2].max(0)
        w, h = x1 - x0, y1 - y0
        x0, x1, y0, y1 = x0 - 0.5 * w, x1 + 0.5 * w, y0 - 0.6 * h, y1 + 0.1 * h
        best = min(best, np.hypot(max(x0 - uv[0], 0, uv[0] - x1), max(y0 - uv[1], 0, uv[1] - y1)))
    return best


def detect_hits(frames, px, cam=None, poses=None, angle_deg=30, reach=8, nms=6, single_flight_px=4.0,
                prox_px=50.0):
    """Hit candidates from a back-view 2D track, robust to the shuttle being hidden at contact.

    1. at every half frame tau, direction of motion from the (up to 3) nearest samples within
       `reach` frames before vs after; candidates turn by > angle_deg (local maxima, NMS).
    2. player gate (if poses given): the shuttle must be within prox_px of a player's
       racket-reach box (keeps 95% of true hits, drops ~40% of false ones on dev).
    3. physics veto (if cam given): if ONE drag+gravity flight explains the samples on both
       sides to < single_flight_px rms, it is not a hit (e.g. the image-space reversal at the
       apex of a clear). Returns sorted fractional frame times."""
    frames = np.asarray(frames, float)
    px = np.asarray(px, float)
    score = []
    for tau in np.arange(frames[0] + 1, frames[-1] - 1 + 1e-9, 0.5):
        pre = np.where((frames < tau) & (frames >= tau - reach))[0][-3:]
        post = np.where((frames > tau) & (frames <= tau + reach))[0][:3]
        if len(pre) < 2 or len(post) < 2:
            continue
        v1 = (px[pre[-1]] - px[pre[0]]) / max(frames[pre[-1]] - frames[pre[0]], 1e-6)
        v2 = (px[post[-1]] - px[post[0]]) / max(frames[post[-1]] - frames[post[0]], 1e-6)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 2 or n2 < 2:  # px/frame: (nearly) static on one side
            continue
        ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1)))
        score.append((ang, tau))
    cand = []
    for ang, tau in sorted(score, reverse=True):
        if ang > angle_deg and all(abs(tau - c) >= nms for c in cand):
            cand.append(tau)
    if poses is not None:
        kept = []
        for tau in cand:
            f = int(round(tau))
            uv = px[np.argmin(np.abs(frames - tau))]
            if 0 <= f < len(poses) and near_player_px(poses[f], uv) <= prox_px:
                kept.append(tau)
        cand = kept
    if cam is None:
        return sorted(cand)
    hits = []
    for tau in cand:
        m = np.abs(frames - tau) <= reach + 2
        f, u = frames[m], px[m]
        if len(f) < 6:
            hits.append(tau)
            continue
        one = RallyFit(cam, f, u, [(f[0], f[-1])], [None], use_continuity=False, use_net=False)
        one.fit()
        rms = np.sqrt(np.mean(np.sum((cam.project(one.position(f)) - u) ** 2, 1)))
        if rms > single_flight_px:
            hits.append(tau)
    return sorted(hits)


def clean_track(frames, px, hits, radius=8.0, min_count=6):
    """Drop static clutter from a 2D shuttle track: detections that recur at (almost) the same
    pixel >= min_count times before the last hit -- spare shuttles on the floor, a white mark,
    the shuttle held before the serve. The final rest after landing is kept (it comes after the
    last hit and is used for landing detection)."""
    frames = np.asarray(frames)
    px = np.asarray(px, float)
    last = max(hits) if len(hits) else frames[-1]
    d = np.linalg.norm(px[:, None] - px[None], axis=2)
    static = (d < radius).sum(1) >= min_count
    keep = ~static | (frames > last)
    return frames[keep], px[keep]


def landing_frame(frames, px, near_px=80.0, min_after=4, frac=0.7):
    """First detection after which the shuttle stays near that spot (on the floor) for the rest of
    the clip: >= frac of the >= min_after later detections within near_px. Tolerates bounces,
    rolling and the odd TrackNet false positive elsewhere; an apex hover fails because the
    shuttle moves away afterwards. None if it never comes to rest in view."""
    frames = np.asarray(frames)
    for i in range(len(frames) - min_after):
        rest = px[i:]
        if (np.linalg.norm(rest - rest[0], axis=1) < near_px).mean() >= frac:
            return frames[i]
    return None


def rest_point(cam, frames, px, lf, still=5.0):
    """Where the shuttle comes to rest after landing at frame lf: the median pixel once it stops
    moving (< still px/frame for two steps; bounces and rolling come first), ray-cast to the floor."""
    frames, px = np.asarray(frames, float), np.asarray(px, float)
    i0 = int(np.searchsorted(frames, lf))
    td = i0
    for i in range(i0, len(frames) - 2):
        if (np.linalg.norm(np.diff(px[i:i + 3], axis=0), axis=1) / np.diff(frames[i:i + 3]) < still).all():
            td = i
            break
    return _ray_point(cam, np.median(px[td:], 0), 0.0)


def segment_shots(frames, hits, px=None):
    """Shots as (t_start, t_end) frame intervals between consecutive hits. The last shot ends at
    the landing (if the shuttle comes to rest in view) or the last detection. Hits up to 10 frames
    before the first detection are kept (the shuttle is often hidden right at the serve)."""
    hits = sorted(h for h in hits if h >= frames[0] - 10)
    end = frames[-1]
    if px is not None:
        lf = landing_frame(frames, px)
        if lf is not None and (not hits or lf > hits[-1] + 3):
            end = lf
    hits = [h for h in hits if h < end - 2]
    if not hits:
        return [(frames[0], end)]
    bounds = list(hits) + [end]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] >= 3]


def nearest_hand(players_at, uv, cam):
    """Hitter = player whose (projected) wrist is closest to the shuttle pixel. Returns 3D hand
    position (racket reach not modelled) or None."""
    c = hand_candidates(players_at, uv, cam, max_px=1e9)
    return c[0]["hand"] if c else None


def hand_candidates(players_at, uv, cam, max_px=300.0):
    """One candidate per player: {'hand': 3D wrist nearest the shuttle in the image,
    'side': 0 near / 1 far half of the court, 'px': image distance}, sorted by image distance,
    keeping players within max_px. In the back view the net player and the back-court player of
    the far side overlap in the image, so image distance alone cannot pick the hitter."""
    out = []
    for p in players_at:
        best, bd = None, 1e9
        for w in p["wrists"]:
            d = np.linalg.norm(cam.project(w[None])[0] - uv)
            if d < bd:
                best, bd = w, d
        if best is not None and bd <= max_px:
            out.append({"hand": best, "side": int(p["foot"][1] > NET_Y), "px": bd})
    return sorted(out, key=lambda c: c["px"])


def _single_flight_rms(cam, f, u):
    one = RallyFit(cam, f, u, [(f[0], f[-1])], [None], use_continuity=False, use_net=False).fit()
    return np.sqrt(np.mean(np.sum((cam.project(one.position(f)) - u) ** 2, 1)))


def split_unexplained(cam, frames, px, shots, max_rms=6.0, min_obs=5, step=1.0, max_rounds=6):
    """Residual-driven segmentation: a 'shot' that one drag+gravity flight cannot explain
    (rms > max_rms px) hides a missed hit. Split it at the time that best explains its samples as
    two flights (each side >= min_obs samples); repeat until every shot is ballistic."""
    frames = np.asarray(frames, float)
    shots = list(shots)
    for _ in range(max_rounds):
        changed = False
        out = []
        for a, b in shots:
            m = (frames >= a) & (frames <= b)
            f, u = frames[m], px[m]
            if len(f) < 2 * min_obs or _single_flight_rms(cam, f, u) <= max_rms:
                out.append((a, b))
                continue
            best = None
            for tau in np.arange(f[min_obs - 1] + 0.5, f[-min_obs] - 0.5 + 1e-9, step):
                l, r = f <= tau, f > tau
                if l.sum() < min_obs or r.sum() < min_obs:
                    continue
                e = _single_flight_rms(cam, f[l], u[l]) ** 2 * l.sum() + _single_flight_rms(cam, f[r], u[r]) ** 2 * r.sum()
                if best is None or e < best[0]:
                    best = (e, tau)
            if best is None:
                out.append((a, b))
                continue
            out += [(a, best[1]), (best[1], b)]
            changed = True
        shots = out
        if not changed:
            break
    return shots


def merge_explained(cam, frames, px, shots, max_rms=6.0):
    """Inverse of split_unexplained: repeatedly merge the adjacent pair of shots that a single
    drag+gravity flight explains best, while it stays under max_rms px (removes false hits)."""
    frames = np.asarray(frames, float)
    shots = list(shots)
    while len(shots) > 1:
        best = None
        for i in range(len(shots) - 1):
            a, c = shots[i][0], shots[i + 1][1]
            m = (frames >= a) & (frames <= c)
            if m.sum() < 4:
                continue
            e = _single_flight_rms(cam, frames[m], px[m])
            if e <= max_rms and (best is None or e < best[0]):
                best = (e, i)
        if best is None:
            break
        i = best[1]
        shots[i:i + 2] = [(shots[i][0], shots[i + 1][1])]
    return shots


def _best_by_fit(cam, frames, px, shot, cands, **kw):
    """cands are one court side's players, sorted by image distance to the shuttle at the hit."""
    if len(cands) <= 1:
        return cands[0]["hand"] if cands else None
    if cands[0]["px"] < 0.5 * cands[1]["px"]:  # unambiguous in the image (typical near side)
        return cands[0]["hand"]
    kw = {**kw, "use_continuity": False}  # a single shot has nothing to be continuous with
    costs = [RallyFit(cam, frames, px, [shot], [c["hand"]], **kw).fit().result.cost for c in cands]
    return cands[int(np.argmin(costs))]["hand"]


VOTE_MARGIN = float(os.environ.get("VOTE_MARGIN", "1.0"))


def fit_rally(cam, frames, px, shots, candidates, land=False, **kw):
    """Full v2 fit. Hitters alternate court sides, so the only discrete unknown is which side
    serves; for each choice, pick each shot's hitter within that side by single-shot fit cost,
    fit the whole rally jointly, and keep the side assignment with the lower joint cost.
    candidates: list over shots of hand_candidates() lists. land: pin the last shot's end to the
    floor when it ends at the detected landing (segment_shots)."""
    lf = landing_frame(frames, px) if land else None
    rest = rest_point(cam, frames, px, lf) if lf is not None and abs(shots[-1][1] - lf) < 0.5 else None
    # candidates may carry "far_p", a hitter model's P(hitter on the far side): a confident vote over the rally's
    # shots fixes the serve side; the fit cost decides only when the vote is split (it can prefer the mirrored rally)
    vote, k = [0.0, 0.0], 0
    for cands in candidates:
        if cands:
            fp = cands[0].get("far_p")
            if fp is not None:
                for s0 in (0, 1):
                    side = s0 if k % 2 == 0 else 1 - s0
                    vote[s0] += (fp if side == 1 else 1 - fp) - 0.5
            k += 1
    order = (0, 1)
    if abs(vote[0] - vote[1]) > VOTE_MARGIN:
        order = (int(vote[1] > vote[0]),)
    best = None
    for s0 in order:
        hands, k = [], 0
        for i, cands in enumerate(candidates):
            if not cands:  # not a hit by a player (e.g. track starts mid-flight): no anchor, no turn
                hands.append(None)
                continue
            side = s0 if k % 2 == 0 else 1 - s0
            k += 1
            same = [c for c in cands if c["side"] == side]
            hands.append(_best_by_fit(cam, frames, px, shots[i], same, **kw))
        f = RallyFit(cam, frames, px, shots, hands, rest=rest, **kw).fit()
        if best is None or f.result.cost < best.result.cost:
            best = f
    return best


def _ray_point(cam, uv, z, fallback_dist=8.0):
    """Point where the pixel's ray reaches height z, or (above the horizon / behind the camera)
    the point fallback_dist metres along the ray."""
    c, d = cam.rays(np.asarray(uv, float).reshape(1, 2))
    d = d[0]
    if abs(d[2]) > 1e-6:
        s_ = (z - c[2]) / d[2]
        if 0 < s_ < 40:
            return c + s_ * d
    return c + fallback_dist * d


class RallyFit:
    def __init__(self, cam, frames, px, shots, hands, sigma_px=4.0, sigma_cont=0.10, sigma_hand=0.6,
                 sigma_net=0.03, sigma_floor=0.05, use_continuity=True, use_net=True, robust=True, rest=None,
                 sigma_land=0.15, net_margin=0.1, use_side=False, net_gap=0.0):
        # dev BD12-14 sweeps: sigma_land 0.15/0.3/0.6; sigma_net 0.01/0.03/0.1 with net_margin -- 0.03 cuts crossings
        # > 15 cm under the tape 36 -> 10 (oracle hits) / 98 -> 18 (detected) at no accuracy cost; 0.01 cost accuracy
        # with detected hits. use_side (hits >= NET_GAP on the hitter's side) hurt with detected hits (wrong sides);
        # net_gap (hits >= NET_GAP off the net, either side) did not help either (dev detected median 0.50 -> 0.52).
        self.cam, self.frames, self.px = cam, np.asarray(frames), np.asarray(px, float)
        self.rest, self.s_land = rest, sigma_land  # floor point the last shot comes down above
        self.net_margin, self.net_gap = net_margin, net_gap  # m above the tape a crossing clears / a hit is off the net
        self.use_side = use_side  # hits on their hitter's side of the net, else just NET_GAP off it (part of use_net)
        self.shots, self.hands = shots, hands
        self.s_px, self.s_cont, self.s_hand = sigma_px, sigma_cont, sigma_hand
        self.s_net, self.s_floor = sigma_net, sigma_floor
        self.use_continuity, self.use_net, self.robust = use_continuity, use_net, robust
        # hitter's court side per shot (+1 far / -1 near): from the hand anchor, else alternating from a known one
        self.sides = [0 if h is None else (1 if h[1] > NET_Y else -1) for h in hands]
        for _ in range(len(hands)):
            for i in range(len(hands)):
                if not self.sides[i]:
                    nb = [-self.sides[k] for k in (i - 1, i + 1) if 0 <= k < len(hands) and self.sides[k]]
                    self.sides[i] = nb[0] if nb else 0
        self.obs = []
        for (a, b) in shots:
            m = (self.frames >= a) & (self.frames <= b)
            self.obs.append((self.frames[m], self.px[m]))

    def _traj(self, params, i, ts_frames):
        p0, v0 = params[6 * i:6 * i + 3], params[6 * i + 3:6 * i + 6]
        return simulate(p0, v0, (np.asarray(ts_frames, float) - self.shots[i][0]) / FPS)

    def residuals(self, params):
        res = []
        n = len(self.shots)
        for i, (a, b) in enumerate(self.shots):
            fr, uv = self.obs[i]
            t_eval = np.r_[fr, b, np.linspace(a, b, N_GRID)]
            Xs = self._traj(params, i, t_eval)
            Xo, Xend, Xgrid = Xs[:len(fr)], Xs[len(fr)], Xs[len(fr) + 1:]
            if len(fr):
                rp = ((self.cam.project(Xo) - uv) / self.s_px).ravel()
                if self.robust:  # soft-L1 on the pixels ONLY; priors must stay quadratic
                    rp = np.sign(rp) * np.sqrt(2 * (np.sqrt(1 + rp * rp) - 1))
                res.append(rp)
            p0 = params[6 * i:6 * i + 3]
            if self.hands[i] is not None:
                res.append((p0 - self.hands[i]) / self.s_hand)
            if self.use_continuity and i + 1 < n:
                res.append((Xend - params[6 * (i + 1):6 * (i + 1) + 3]) / self.s_cont)
            res.append(np.minimum(Xgrid[:, 2], 0) / self.s_floor)  # never below the floor
            if self.rest is not None and i == n - 1:
                res.append((Xend[:2] - self.rest[:2]) / self.s_land)
            if self.use_net:  # fixed slots per shot (least_squares needs a constant length)
                # the hit is on the hitter's side, clear of the net (a hand anchor is loose enough to slide a
                # hit onto the net plane, where the shuttle would then be struck below the tape)
                side = self.sides[i] if self.use_side else 0
                gap = side * (p0[1] - NET_Y) if side else abs(p0[1] - NET_Y)  # no side known: either side will do
                res.append(np.atleast_1d(min(gap - self.net_gap, 0.0) / self.s_net))
                clear = 0.0
                dy = Xgrid[:, 1] - NET_Y
                j = np.flatnonzero(dy[:-1] * dy[1:] <= 0)
                if len(j):  # crosses the net plane (twice if a missed hit merged two flights): the lowest
                    w = np.where(dy[j] != dy[j + 1], dy[j] / (dy[j] - dy[j + 1] + 1e-12), 0.0)  # height there
                    clear = min(np.min((1 - w) * Xgrid[j, 2] + w * Xgrid[j + 1, 2]) - POST_H - self.net_margin, 0.0)
                res.append(np.atleast_1d(clear / self.s_net))
        return np.concatenate(res)

    def init_params(self):
        """Physically consistent start: each shot is a drag+gravity flight from its hitter's hand
        to the next hitter's hand (last shot: to where its last detection ray-casts to the floor),
        with the launch velocity solved as a two-point boundary problem."""
        n = len(self.shots)
        starts = []
        for i, (a, b) in enumerate(self.shots):
            if self.hands[i] is not None:
                starts.append(np.array(self.hands[i], float))
            else:
                fr, uv = self.obs[i]
                p = _ray_point(self.cam, uv[0], 1.5) if len(fr) else np.array([3.05, L / 2, 1.5])
                starts.append(p)
        params = []
        for i, (a, b) in enumerate(self.shots):
            T = max((b - a) / FPS, 0.1)
            if i + 1 < n:
                end = starts[i + 1]
            else:
                fr, uv = self.obs[i]
                end = (_ray_point(self.cam, uv[-1], 0.0) if len(fr)
                       else np.array([starts[i][0], L - starts[i][1], 0.0]))
            v = (end - starts[i]) / T
            v[2] += 0.5 * 9.81 * T
            sol = least_squares(lambda vv: simulate(starts[i], vv, [T])[0] - end, v, max_nfev=50)
            params.append(np.r_[starts[i], sol.x])
        return np.concatenate(params)

    def fit(self, x0=None):
        x0 = self.init_params() if x0 is None else x0
        r = least_squares(self.residuals, x0, x_scale="jac", max_nfev=300)
        self.params = r.x
        self.result = r
        return self

    def position(self, t_frames, rest_until=None):
        """3D position at arbitrary (fractional) back-view frame times; NaN outside the shots.
        After the last shot, the shuttle rests where it landed (up to rest_until)."""
        t_frames = np.asarray(t_frames, float)
        out = np.full((len(t_frames), 3), np.nan)
        for i, (a, b) in enumerate(self.shots):
            m = (t_frames >= a) & (t_frames <= b)
            if m.any():
                out[m] = self._traj(self.params, i, t_frames[m])
        if rest_until is not None:
            b = self.shots[-1][1]
            land = self._traj(self.params, len(self.shots) - 1, [b])[0]
            land[2] = 0.0
            m = (t_frames > b) & (t_frames <= rest_until)
            out[m] = land
        return out
