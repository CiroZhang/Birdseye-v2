"""Shot-by-shot 3D shuttle from the detected hits: every shot is one free flight (gravity + air drag) from the hitter's
racket to the next hitter's racket, fitted to what every camera sees.

Given the rally's hits and hitters (tools/hit_decode.py HD_HITS: [[TV frame, multi-camera track]]), the unknowns are
the 3D contact point of every hit and the launch velocity of every shot (the last shot flies on until the track ends).
Residuals, robust (soft L1):
  - every camera's shuttle detection against the simulated flight (TV and side refined tracks; player cameras' streak
    peaks, the peak nearest the flight once a first fit exists),
  - a shot ends at the next contact point (the flight must arrive where the next hitter hits it),
  - each contact point within racket reach of the hitter's 3D wrists (tools/mv_wrist.py; without wrists, within reach
    of the hitter's position on the floor) and on the hitter's side of the net,
  - a shot between the two sides passes over the net (above the tape where it crosses the net plane),
  - the shuttle never below the floor.
Output outputs/cctv/<SF_OUT or shot_fit>/<uid>.json {frames, P (per TV frame), shot (index per frame), contacts
[[frame, track, x, y, z]], px_rms {camera: px}, n_obs {camera: count}, res (per frame: worst camera px)}.
usage: python tools/shot_fit.py <hits json> uid ..."""
import json, os, sys

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tools"))
import body_mv as BM  # noqa: E402
import mv_people as MP  # noqa: E402
import mv_wrist as MW  # noqa: E402
from birdseye_v2 import shuttle3d as S3  # noqa: E402

FPS = 25.0
NET_Y, NET_H, W = 6.7, 1.55, 6.1
SIG_PX = {"tv": 4.0, "side": 4.0, "p1": 15.0, "p2": 15.0}
PX_SCALE = 3.0          # soft-L1 scale in sigmas: detections further off than this count less (wrong detections)
SIG_LINK = 0.03         # m: a shot ends at the next contact point
REACH = (0.42, 0.62)    # m: racket head centre from the wrist holding it (grip 7 cm past the wrist + 0.47 m to the head)
SIG_REACH = float(os.environ.get("SF_SIG_REACH", "0.05"))
BODY = os.environ.get("SF_BODY", "body_mvt")   # the drawn bodies (tools/body_mv.py): the hitter's wrists come from them
FLOOR_REACH = 1.7       # m: contact from the hitter's floor point when no wrist is seen
SIG_NET = 0.05
MAX_LAST = 60           # frames: the last shot (after the final hit) is followed at most this long
PCAM_GATE = 30.0        # px: player-camera peak taken when this close to the first fit's projection
OUT = os.path.join(ROOT, "outputs", "cctv", os.environ.get("SF_OUT", "shot_fit"))


class Rally:
    def __init__(s, uid, hits):
        import cctv_tri as CT
        import cctv_teach as CTE
        s.uid = uid; s.hits = sorted((int(f), int(k)) for f, k in hits)
        s.wr = MW.Wrists(uid); s.pe = s.wr.pe; s.m = s.wr.m; s.T = len(s.m["foot"])
        s.obs = []                                     # (camera dict, TV time, pixel (2,), view)
        for v, tr in s.wr.shut.items():                # refined TV / side tracks, keyed by TV frame
            for f, px in tr.items():
                k = s.pe.frame_of(v, f); cam = s.pe.views[v][0](k) if k is not None else None
                if cam is not None and 0 <= f < s.T:
                    s.obs.append((cam, float(f), px, v))
        s.pcam = []                                    # (camera dict, TV time, peaks (m,2), view)
        for v in ("p1", "p2"):
            p = os.path.join(ROOT, "outputs", "cctv_" + v, "shuttle_streak_m", uid + ".json")
            if v not in s.pe.views or not os.path.exists(p):
                continue
            S = json.load(open(p)); camf = s.pe.views[v][0]
            for k, pk in zip(S["frames"], S["peaks"]):
                cam = camf(k)
                if pk and cam is not None:
                    s.pcam.append((cam, k - CTE.PCAM_SYNC[v], np.array(pk, float)[:, :2], v))
        s.end = int(max([o[1] for o in s.obs if o[1] >= s.hits[-1][0]] or [s.hits[-1][0] + 10]))
        bp = os.path.join(ROOT, "outputs", "cctv", BODY, uid + ".npz")
        s.body = np.load(bp) if os.path.exists(bp) else None
        K = int(s.m["K"]); sd = [np.nanmedian(s.m["foot"][:, q, 1]) > NET_Y for q in range(K)]
        s.slot = {q: ("top" if sd[q] else "bottom") + str(sum(1 for j in range(q) if sd[j] == sd[q])) for q in range(K)}
        rp = os.path.join(ROOT, "outputs", "cctv", os.environ.get("SF_RACKET", "racket_mvt"), uid + ".npz")
        rz = np.load(rp) if os.path.exists(rp) else None           # each player's racket hand (racket box votes)
        s.hand = {q: int(rz[s.slot[q] + "_hand"]) for q in range(K) if rz is not None and s.slot[q] + "_hand" in rz.files}

    def side(s, k, f):
        y = s.m["foot"][min(f, s.T - 1), k, 1]
        return None if not np.isfinite(y) else y > NET_Y

    def reach_targets(s, f, k):
        """hitter k's wrists (3D) at frame f: the drawn body's (both hands; the racket hand is not known here), else the
        cameras' triangulated wrists; else its floor point"""
        W3 = []
        if s.body is not None and s.slot.get(k) in s.body.files:
            fr = {int(x): i for i, x in enumerate(s.body["frames"])}
            if f in fr:
                hands = [s.hand[k]] if k in s.hand else [20, 21]
                W3 = [w for w in s.body[s.slot[k]][fr[f], hands].astype(float) if np.isfinite(w).all()]
        if not W3:
            W3 = [X for t in (f, f - 1, f + 1) if 0 <= t < s.T for X, rms in s.wr.wrists3d(t, k) if rms <= 0.3]
        foot = s.m["foot"][min(f, s.T - 1), k]
        return W3, (foot if np.isfinite(foot).all() else None)


def shot_times(R):
    """per shot: (start frame, end frame) -- the last shot ends where the cameras last see the shuttle"""
    H = [f for f, _ in R.hits]
    return [(H[i], H[i + 1] if i + 1 < len(H) else min(R.end, H[-1] + MAX_LAST)) for i in range(len(H))]


def landing(C, V, a, b):
    """first frame in [a, b] where the last shot's flight reaches the floor (the rally is over there)"""
    ts = np.arange(a, b + 1); Z = flight(C, V, a, ts)[:, 2]
    i = np.where(Z <= 0.02)[0]
    return int(ts[i[0]]) if len(i) else b


def flight(C, V, t0, ts):
    return S3.simulate(C, V, (np.asarray(ts, float) - t0) / FPS)


def robust(r, c=PX_SCALE):
    """soft-L1 on camera residuals only (as least_squares' loss, but leaving the physics terms plain least squares)"""
    r = np.asarray(r, float)
    return np.sign(r) * np.sqrt(2 * c * c * (np.sqrt(1 + (r / c) ** 2) - 1))


def residuals(x, R, shots, obs_by_shot, targets, sides):
    n = len(shots)
    C = x[:3 * n].reshape(n, 3); V = x[3 * n:6 * n].reshape(n, 3)
    res = []
    for i, (a, b) in enumerate(shots):
        ob = obs_by_shot[i]
        ts = [o[1] for o in ob] + [b] + list(np.linspace(a, b, 12))
        X = flight(C[i], V[i], a, ts)
        for (cam, t, px, v), Xi in zip(ob, X[:len(ob)]):
            uv, dep = BM.project(cam, Xi[None])
            r = (uv[0] - px) / SIG_PX[v] if dep[0] > 0.3 else np.array([20.0, 20.0])
            res.extend(robust(r))                                      # only detections are robust: the physics is not
        if i + 1 < n:
            res.extend((X[len(ob)] - C[i + 1]) / SIG_LINK)            # the flight arrives at the next contact
        G = X[len(ob) + 1:]
        if i + 1 < n:                                                  # never below the floor (the last shot lands)
            res.extend(np.minimum(G[:, 2], 0.0) / 0.05)
        if i + 1 < n and sides[i] is not None and sides[i + 1] is not None and sides[i] != sides[i + 1]:
            Y = G[:, 1] - NET_Y; cross = np.where(np.sign(Y[:-1]) != np.sign(Y[1:]))[0]
            if len(cross):                                             # height where it crosses the net plane
                j = cross[0]; w = Y[j] / (Y[j] - Y[j + 1]); z = G[j, 2] + w * (G[j + 1, 2] - G[j, 2])
                res.append(min(z - NET_H - 0.05, 0.0) / SIG_NET)
            else:
                res.append(-1.0 / SIG_NET)                             # never reaches the other side
    for i in range(n):                                                 # contacts within reach of the hitter
        W3, foot = targets[i]
        if W3:
            d = min(np.linalg.norm(C[i] - w) for w in W3)
            res.append((max(d - REACH[1], 0.0) + min(d - REACH[0], 0.0)) / SIG_REACH)
        elif foot is not None:
            d = np.linalg.norm(C[i, :2] - foot)
            res.append(max(d - FLOOR_REACH, 0.0) / SIG_REACH)
        else:
            res.append(0.0)
        if sides[i] is not None:                                       # on the hitter's side of the net
            g = (C[i, 1] - NET_Y) if sides[i] else (NET_Y - C[i, 1])
            res.append(min(g - 0.1, 0.0) / SIG_NET)
        else:
            res.append(0.0)
        res.append(min(C[i, 2] - 0.1, 0.0) / 0.05)
    return np.array(res, float)


def sparsity(shots, obs_by_shot, sides):
    """which parameters each residual of residuals() depends on: shot i's rows on its contact and launch velocity (its link
    also on the next contact), each contact's rows on that contact alone -- lets least_squares group the finite differences"""
    from scipy.sparse import lil_matrix
    n = len(shots); rows = []
    cv = lambda i: list(range(3 * i, 3 * i + 3)) + list(range(3 * n + 3 * i, 3 * n + 3 * i + 3))
    for i in range(n):
        rows += [cv(i)] * (2 * len(obs_by_shot[i]))
        if i + 1 < n:
            rows += [cv(i) + list(range(3 * i + 3, 3 * i + 6))] * 3 + [cv(i)] * 12
            if sides[i] is not None and sides[i + 1] is not None and sides[i] != sides[i + 1]:
                rows += [cv(i)]
    for i in range(n):
        rows += [list(range(3 * i, 3 * i + 3))] * 3
    A = lil_matrix((len(rows), 6 * n), dtype=int)
    for r, cols in enumerate(rows):
        A[r, cols] = 1
    return A


def initial(R, shots, targets, sides):
    """contact points at the hitter's wrists (or above its floor point, toward the net); velocities that reach the next"""
    n = len(shots); C = np.zeros((n, 3))
    for i, (W3, foot) in enumerate(targets):
        if W3:
            w = min(W3, key=lambda X: -X[2]); C[i] = w + [0, (-0.3 if sides[i] else 0.3) if sides[i] is not None else 0, 0.3]
        elif foot is not None:
            C[i] = [foot[0], foot[1] + ((-0.5 if sides[i] else 0.5) if sides[i] is not None else 0), 2.0]
        else:
            C[i] = [W / 2, NET_Y + (2.0 if (sides[i] is None or sides[i]) else -2.0), 2.0]
    V = np.zeros((n, 3))
    for i, (a, b) in enumerate(shots):
        tgt = C[i + 1] if i + 1 < n else C[i] + [0, (-6 if C[i, 1] > NET_Y else 6), -C[i, 2]]
        dt = max(b - a, 1) / FPS
        f = lambda v: S3.simulate(C[i], v, [dt])[0] - tgt
        V[i] = least_squares(f, (tgt - C[i]) / dt + [0, 0, 4.9 * dt], method="lm").x
    return np.r_[C.ravel(), V.ravel()]


def fit(uid, hits):
    """fit with the decoded hitters, then in doubles try each hit's partner as the hitter and keep it where the flight
    explains the cameras and the racket contact better (the shuttle can only meet the true hitter's racket)"""
    out = solve(uid, hits)
    if not os.environ.get("SF_SWAP", "1") == "1":
        return out
    R = Rally(uid, hits); K = int(R.m["K"])
    if K < 4:
        return out
    cur = [list(h) for h in sorted(hits)]
    for i, (f, k) in enumerate(list(cur)):
        side = R.side(k, f)
        mates = [q for q in range(K) if q != k and R.side(q, f) == side]
        for q in mates:
            trial = [list(h) for h in cur]; trial[i][1] = q
            o2 = solve(uid, trial)
            if o2["cost"] < out["cost"] - SWAP_MARGIN:
                out, cur = o2, trial
                out.setdefault("swaps", []).append([f, k, q])
    return out


SWAP_MARGIN = 2.0       # cost units (half sum of squared sigmas): a swapped hitter must explain clearly more


def solve(uid, hits):
    R = Rally(uid, hits)
    shots = shot_times(R)
    targets = [R.reach_targets(f, k) for f, k in R.hits]
    sides = [R.side(k, f) for f, k in R.hits]
    by_shot = lambda obs: [[o for o in obs if a <= o[1] < b or (i == len(shots) - 1 and a <= o[1] <= b)] for i, (a, b) in enumerate(shots)]
    x0 = initial(R, shots, targets, sides)
    obs_by = by_shot(R.obs)
    fun = lambda x, ob: residuals(x, R, shots, ob, targets, sides)
    sol = least_squares(fun, x0, args=(obs_by,), max_nfev=200, x_scale="jac", jac_sparsity=sparsity(shots, obs_by, sides))
    # second pass: add the player cameras' peaks nearest the first fit
    n = len(shots); C = sol.x[:3 * n].reshape(n, 3); V = sol.x[3 * n:].reshape(n, 3)
    shots[-1] = (shots[-1][0], landing(C[-1], V[-1], *shots[-1]))   # after the landing the shuttle is picked up
    extra = []
    for cam, t, pk, v in R.pcam:
        i = next((j for j, (a, b) in enumerate(shots) if a <= t < b or (j == n - 1 and a <= t <= b)), None)
        if i is None:
            continue
        X = flight(C[i], V[i], shots[i][0], [t])[0]
        uv, dep = BM.project(cam, X[None])
        if dep[0] <= 0.3:
            continue
        d = np.linalg.norm(pk - uv[0], axis=1); j = int(np.argmin(d))
        if d[j] < PCAM_GATE:
            extra.append((cam, t, pk[j], v))
    obs_by = by_shot(R.obs + extra)
    sol = least_squares(fun, sol.x, args=(obs_by,), max_nfev=200, x_scale="jac", jac_sparsity=sparsity(shots, obs_by, sides))
    C = sol.x[:3 * n].reshape(n, 3); V = sol.x[3 * n:].reshape(n, 3)
    frames = np.arange(shots[0][0], shots[-1][1] + 1)
    P = np.zeros((len(frames), 3)); shot = np.zeros(len(frames), int)
    for i, (a, b) in enumerate(shots):
        m = (frames >= a) & ((frames < b) | (i == n - 1))
        P[m] = flight(C[i], V[i], a, frames[m]); shot[m] = i
    # fit quality per camera and per frame
    err = {v: [] for v in SIG_PX}; worst = np.zeros(len(frames))
    for i, ob in enumerate(obs_by):
        for cam, t, px, v in ob:
            X = flight(C[i], V[i], shots[i][0], [t])[0]
            e = float(np.linalg.norm(BM.project(cam, X[None])[0][0] - px)); err[v].append(e)
            j = int(round(t)) - frames[0]
            if 0 <= j < len(worst):
                worst[j] = max(worst[j], e / SIG_PX[v])
    out = {"uid": uid, "frames": frames.tolist(), "P": np.round(P, 4).tolist(), "shot": shot.tolist(),
           "contacts": [[f, k, *np.round(C[i], 3).tolist()] for i, (f, k) in enumerate(R.hits)],
           "px_med": {v: round(float(np.median(e)), 2) for v, e in err.items() if e},
           "n_obs": {v: len(e) for v, e in err.items()}, "res": np.round(worst, 2).tolist(), "cost": float(sol.cost)}
    return out


def main(hits_path, uids):
    H = json.load(open(hits_path)); os.makedirs(OUT, exist_ok=True)
    for u in uids:
        try:
            o = fit(u, H[u])
            json.dump(o, open(os.path.join(OUT, u + ".json"), "w"))
            print(u, "shots", len(o["contacts"]), "px median", o["px_med"], "obs", o["n_obs"], flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(u, "FAILED", repr(e), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
