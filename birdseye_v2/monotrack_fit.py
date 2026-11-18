"""MonoTrack's 3D shuttle reconstruction (Liu & Wang, CVPRW 2022), as a baseline on our inputs.

The per-shot fit is MonoTrack's own RallyReconstructor.reconstruct_one_hit / reconstruct
(third_party/ext/monotrack/python/ai-badminton/src/ai_badminton/rally_reconstructor.py, github.com/jhwang7628/monotrack),
copied with the optimisation unchanged: per shot, a start position near the hitter's feet at 1.7 m (or the previous
shot's end when it reprojects within 40 px), initial velocity and drag coefficient fitted by SLSQP so that the
gravity + quadratic-drag flight reprojects onto the 2D track (L6 norm), with penalties for landing outside the court
and for the flight's end drifting from the next hitter's feet. Only the inputs are ours, shared with the other
baselines: the camera (our calibration, as MonoTrack's 3x4 projection), the 2D shuttle track, the hit frames and the
players' foot points on the court (our 2D pose + calibration; MonoTrack uses MMPose + its own court detector)."""
import numpy as np
import scipy.optimize

W, L = 6.1, 13.4


def cam_proj(cam):
    P = cam.K @ np.c_[cam.R, np.asarray(cam.tvec, float).reshape(3, 1)]
    return P / P[2, 3]


def reconstruct_one_hit(camProj, X, Y, s2d, e2d, T, s3d=None, fps=25, fr_adjust=30. / 25):
    """MonoTrack's reconstruct_one_hit, verbatim apart from reading the track from arrays X, Y (0 or NaN = missing)."""
    substeps = 30
    g = np.array([0, 0, -9.8])
    height_guess = 1.7
    xg = np.array(s2d.tolist() + [height_guess])
    if s3d is not None:
        xg = np.array(s3d)
    N = T[1] - T[0]
    td = N / float(fps)
    vg = np.array(((e2d - s2d) / td).tolist() + [9.8 * td / 2 - 1. / td])
    Cg = 0.2
    initg = np.concatenate([xg, vg, np.array([Cg])])
    bounds = [(0, 6.1), (0, 13.4 / 2), (0.1, 6)] + [(-150, 150)] + [(0, 150)] + [(-150, 150)] + [(0, 0.4)]
    if s2d[1] > 13.4 / 2:
        bounds[1] = (13.4 / 2, 13.4)
        bounds[4] = (-150, 0)
    norm_scale = np.linalg.norm(camProj[:3, :3], ord=2)

    def get_trajectory(p):
        x = np.array(p[:3], float)
        v = np.array(p[3:-1], float)
        C = np.array(p[-1])
        dt = fr_adjust * 1. / (fps * substeps)
        res = []
        for t in range(substeps * N + 1):
            v += dt * (g - C * np.linalg.norm(v) * v)
            if t % substeps == 0:
                res.append(np.array(x))
            x += dt * v
        return res

    def f(p):
        x = np.array(p[:3], float)
        if s3d is not None:
            x = np.array(s3d, float)
        v = np.array(p[3:-1], float)
        C = np.array(p[-1])
        loss = 0
        dt = fr_adjust / (fps * substeps)
        pord = 6
        drift_loss, out_loss = 0, 0
        tid = T[0]
        for t in range(1, substeps * N + 1):
            v += dt * (g - C * np.linalg.norm(v) * v)
            if t % substeps == 0:
                q = camProj[:, :3] @ x + camProj[:, 3]
                z = q[:2] / q[2]
                if 0 <= tid < len(X) and X[tid] != 0 and X[tid] == X[tid]:
                    z_ = np.array([X[tid], Y[tid]])
                    loss += np.linalg.norm(z - z_, ord=pord) ** pord
                tid += 1
                if tid == T[1]:
                    xc, vc = np.array(x), np.array(v)
                    n_it = 0
                    while xc[2] > 0 and n_it < 10000:
                        xc += dt * vc
                        vc += dt * (g - C * np.linalg.norm(vc) * vc)
                        n_it += 1
                    out_loss += max(0 - xc[0], xc[0] - 6.1, 0) ** pord
                    out_loss += max(0 - xc[1], xc[1] - 13.4, 0) ** pord
                    out_loss += max(0 - xc[2], xc[2] - 3., 0) ** pord
                    drift_loss += np.linalg.norm(x[:2] - e2d) ** pord
            x += dt * v
        pinv = 1. / pord
        return loss ** pinv + (norm_scale * drift_loss) ** pinv

    res = scipy.optimize.minimize(f, initg, bounds=bounds, method='SLSQP')
    return np.array(get_trajectory(res.x))


def project(camProj, p):
    Z = camProj @ np.r_[p, 1.0]
    return Z[:2] / Z[2]


def reconstruct(cam, frames, px, hits, foot_at, fps=25):
    """MonoTrack's RallyReconstructor.reconstruct on our inputs.
    frames, px: the 2D track; hits: sorted integer hit frames (the first one starts the first shot);
    foot_at(frame) -> (x, y) court position of the player hitting / receiving at that frame (None if unknown).
    Returns position(tq) -> (len(tq), 3), NaN outside the reconstructed shots."""
    P = cam_proj(cam)
    n = int(max(frames.max(), max(hits)) + 2)
    X, Y = np.zeros(n), np.zeros(n)
    fi = np.round(frames).astype(int)
    X[fi], Y[fi] = px[:, 0], px[:, 1]
    fr_adjust = 30. / fps  # as MonoTrack's reconstruct()
    hits = [int(h) for h in hits]
    traj = {}
    s3d = None
    for i in range(len(hits) - 1):
        a, b = hits[i], hits[i + 1]
        if b - a < 2:
            s3d = None
            continue
        s2d, e2d = foot_at(a), foot_at(b)
        if s2d is None or e2d is None:
            s3d = None
            continue
        try:
            tr = reconstruct_one_hit(P, X, Y, np.asarray(s2d, float), np.asarray(e2d, float), [a, b], s3d, fps, fr_adjust)
        except Exception:
            s3d = None
            continue
        last = project(P, tr[-1])
        s3d = tr[-1] if (X[b] != 0 and np.linalg.norm(last - np.array([X[b], Y[b]])) < 40) else None
        for k in range(b - a):
            traj[a + k] = tr[k]

    def position(tq):
        tq = np.asarray(tq, float)
        out = np.full((len(tq), 3), np.nan)
        for j, t in enumerate(tq):
            k0 = int(np.floor(t))
            if k0 in traj and (k0 + 1) in traj:
                w = t - k0
                out[j] = (1 - w) * traj[k0] + w * traj[k0 + 1]
            elif int(round(t)) in traj:
                out[j] = traj[int(round(t))]
        return out
    return position
