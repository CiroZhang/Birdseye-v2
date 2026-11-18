"""Hit picking against the hand answer key: candidates from every cue -> learned hit / not-hit and who-hit scores ->
timing snap -> evaluation (leave-one-rally-out, so every score is on a rally the models never saw).

Candidates: outputs of the scratch candidate merger (sound, sound+vision, teacher hits, shuttle turns), ~99% recall.
Per candidate (TV frame f) features:
  sound / sound+vision / vision-only hit probabilities (max over f+-2 frames), teacher hit within 3 frames, shuttle seen
  and turning in TV and side, gaps to the neighbouring candidates, teacher shuttle height / distance to the net / to the
  nearest player's feet, and per camera (TV, side, player cam 1, player cam 2) the strongest swing of any player:
  wrist speed relative to body height and wrist raised above the shoulder, over f-3..f+1.
Per (candidate, tracked player) features for the hitter: the same swing features for the person each camera shows at
that player's feet (feet projected into the camera), the player's distance to the teacher's shuttle, same side of the
net as the shuttle, and that player's TV-ray wrist speed.
usage: python tools/hit_pick.py <answer key json> <candidates json>"""
import json, os, sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import body_mv as BM  # noqa: E402
from cctv_teach import PCAM_SYNC  # noqa: E402
import importlib.util  # noqa: E402

NET_Y = 6.7
WIN = (-3, 1)


def bridge_feet(F, max_gap=50):
    """fill short tracking dropouts of each player's feet (same person before and after) by linear interpolation"""
    F = F.copy()
    for p in range(F.shape[1]):
        ok = np.flatnonzero(np.isfinite(F[:, p, 0]))
        for a, b in zip(ok[:-1], ok[1:]):
            if 1 < b - a <= max_gap:
                t = (np.arange(a + 1, b) - a) / (b - a)
                F[a + 1:b, p] = (1 - t)[:, None] * F[a, p] + t[:, None] * F[b, p]
    return F


def load_kp(p):
    if not os.path.exists(p):
        return None
    z = np.load(p); c = np.r_[0, np.cumsum(z["counts"])]
    return [z["kp"][c[i]:c[i + 1]] for i in range(len(z["counts"]))]


def models():
    out = {}
    for mode in ("audio", "av", "vision"):
        os.environ["AH_MODE"] = mode
        spec = importlib.util.spec_from_file_location("ah_" + mode, os.path.join(ROOT, "tools", "audio_hits.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        import torch
        net = m.Net().to(m.DEV); net.load_state_dict(torch.load(os.path.join(m.OUT, "model.pt"))); net.eval()
        out[mode] = (m, net)
    return out


def person_feats(P0, P1):
    """per person in P0 (n,17,3): (wrist speed / height, wrist raise / height, ankle point, height) using P1 for speed"""
    res = []
    for p in P0:
        g = p[:, 2] > 0.3
        if g.sum() < 6:
            continue
        hgt = np.ptp(p[g, 1]) + 1
        hip = p[[11, 12], :2].mean(0); ank = p[[15, 16], :2].mean(0) if (p[[15, 16], 2] > 0.3).any() else p[g, :2][np.argmax(p[g, 1])]
        sp = 0.0
        if P1 is not None and len(P1):
            d = [np.linalg.norm(q[[11, 12], :2].mean(0) - hip) for q in P1]
            j = int(np.argmin(d))
            if d[j] < 0.3 * hgt:
                q = P1[j]
                for w in (9, 10):
                    if p[w, 2] > 0.3 and q[w, 2] > 0.3:
                        sp = max(sp, np.linalg.norm(q[w, :2] - p[w, :2]) / hgt)
        rz = max([(p[s, 1] - p[w, 1]) / hgt for s, w in ((5, 9), (6, 10)) if p[s, 2] > 0.3 and p[w, 2] > 0.3] or [0.0])
        res.append((sp, rz, ank, hgt))
    return res


def mv_tracks(uid, T):
    """multi-camera tracks as (foot (T,4,2), TV keypoint rays (T,4,17,3), their confidences (T,4,17)) in slot layout"""
    import mv_people as MP
    from birdseye_v2.court import Camera
    m = np.load(f"{ROOT}/outputs/cctv/mv_track/{uid}.npz"); K = int(m["K"]); n = min(T, len(m["foot"]))
    foot = np.full((T, 4, 2), np.nan); foot[:n, :K] = m["foot"][:n]
    kd = np.zeros((T, 4, 17, 3)); kc = np.zeros((T, 4, 17))
    cam = BM.static_cam(Camera.from_dict(json.load(open(f"{ROOT}/outputs/cctv/calib_bc/{uid}.json"))["cam"]))
    K_tv = BM.load_kp(f"{ROOT}/outputs/cctv/pose1280/{uid}.npz")
    for t in range(n):
        for k in range(K):
            i = int(m["tv_idx"][t, k])
            if i >= 0 and t < len(K_tv) and i < len(K_tv[t]):
                kp = K_tv[t][i]; kd[t, k] = MP.rays(cam, kp[:, :2]); kc[t, k] = kp[:, 2]
    return bridge_feet(foot), kd, kc


class Rally:
    def __init__(s, uid, M):
        s.uid = uid
        s.cams = {c[0]: c for c in BM.cameras(uid)}
        tri = json.load(open(os.path.join(ROOT, "outputs", "cctv", "tri", uid + ".json")))
        s.k_of = {"tv": lambda t: t, "side": lambda t, sh=tri["shift"]: t + sh,
                  "p1": lambda t: int(round(t + PCAM_SYNC["p1"])), "p2": lambda t: int(round(t + PCAM_SYNC["p2"]))}
        s.K = {"tv": load_kp(f"{ROOT}/outputs/cctv/pose1280/{uid}.npz"), "side": load_kp(f"{ROOT}/outputs/cctv_side/pose1280/{uid}.npz"),
               "p1": load_kp(f"{ROOT}/outputs/cctv_p1/pose/{uid}.npz"), "p2": load_kp(f"{ROOT}/outputs/cctv_p2/pose/{uid}.npz")}
        ob = np.load(f"{ROOT}/outputs/rally_obs/cctv/{uid}.npz"); s.foot = bridge_feet(ob["foot"]); s.kd = ob["kp_dir"]; s.kc = ob["kp_conf"]
        s.wr = None
        if os.environ.get("HP_TRACK") == "mv":            # multi-camera player tracks (tools/mv_track.py)
            s.foot, s.kd, s.kc = mv_tracks(uid, s.kd.shape[0])
            import mv_wrist
            s.wr = mv_wrist.Wrists(uid); s.wr_cache = {}
        o = json.load(open(os.path.join(ROOT, "outputs", "cctv", os.environ.get("HP_TEACH", "teach"), uid + ".json")))
        s.tfr = np.array(o["frames"], float); s.tP = np.array(o["P"], float); s.thits = np.array(o["hits"], float)
        s.prob = {}
        for mode, (m, net) in M.items():
            os.environ["AH_MODE"] = mode
            s.prob[mode] = m.prob(net, m.feats(uid))
            if mode == "av":
                s.vis = m.vis(uid, s.prob[mode].shape[0])
        s.pf = {}
        vp = os.path.join(ROOT, "outputs", "cctv", os.environ.get("HV_DIR", "hitvid_local"), "key_probs", uid + ".npz")
        s.vid = np.load(vp) if os.path.exists(vp) else None     # video hit model: per-frame hit and far-side probability

    def cam_dict(s, v, k):
        if v not in s.cams:
            return None
        return s.cams[v][1](k)

    def persons(s, v, k):
        key = (v, k)
        if key not in s.pf:
            K = s.K[v]
            s.pf[key] = person_feats(K[k], K[k + 1] if k + 1 < len(K) else None) if K is not None and 0 <= k < len(K) else []
        return s.pf[key]

    def P(s, f):
        if not len(s.tfr):
            return None
        return np.array([np.interp(f, s.tfr, s.tP[:, a]) for a in range(3)])

    def cand_feats(s, f, prevgap, nextgap, src):
        x = []
        for mode in ("audio", "av", "vision"):
            p = s.prob[mode]; a, b = max(0, (f - 2) * 8), min(len(p), (f + 3) * 8)
            x.append(float(p[a:b].max()) if b > a else 0.0)
        v = s.vis; a, b = max(0, (f - 2) * 8), min(v.shape[1], (f + 3) * 8)
        w = v[:, a:b] if b > a else np.zeros((v.shape[0], 1))
        x += [w[0].max(), w[3].max(), w[4].max(), w[7].max(), w[8].max(), w[10].min()]
        x += [float(np.min(np.abs(s.thits - f))) if len(s.thits) else 99.0]
        x += [min(prevgap, 60), min(nextgap, 60)] + [float(c in src) for c in "SFTU"]
        P = s.P(f)
        if P is None:
            x += [0, 0, 9]
        else:
            ft = s.foot[min(f, len(s.foot) - 1)]
            d = np.linalg.norm(ft - P[:2], axis=1); d = d[np.isfinite(d)]
            x += [P[2], abs(P[1] - NET_Y), float(d.min()) if len(d) else 9.0]
        for v in ("tv", "side", "p1", "p2"):
            sp = rz = 0.0
            for t in range(f + WIN[0], f + WIN[1] + 1):
                for q in s.persons(v, s.k_of[v](t)):
                    sp, rz = max(sp, q[0]), max(rz, q[1])
            x += [sp, rz]
        return x

    def player_feats(s, f, p):
        """features of tracked player p as the hitter of a hit at TV frame f"""
        x = []
        ft = s.foot[min(f, len(s.foot) - 1), p]
        ok = np.isfinite(ft).all()
        for v in ("tv", "side", "p1", "p2"):
            sp = rz = 0.0; seen = 0.0
            if ok:
                for t in range(f + WIN[0], f + WIN[1] + 1):
                    k = s.k_of[v](t); cam = s.cam_dict(v, k)
                    if cam is None:
                        continue
                    uv, dep = BM.project(cam, np.r_[ft, 0.0][None])
                    if dep[0] <= 0.3:
                        continue
                    best = None
                    for q in s.persons(v, k):
                        dd = np.linalg.norm(q[2] - uv[0]) / q[3]
                        if dd < 0.5 and (best is None or dd < best[0]):
                            best = (dd, q)
                    if best is not None:
                        seen = 1.0; sp, rz = max(sp, best[1][0]), max(rz, best[1][1])
            x += [sp, rz, seen]
        P = s.P(f)
        if P is None or not ok:
            x += [9.0, 0.5]
        else:
            x += [float(np.linalg.norm(ft - P[:2])), float((ft[1] < NET_Y) == (P[1] < NET_Y))]
        # TV-ray wrist speed of this player (degrees per frame)
        sp = 0.0
        for t in range(max(1, f - 3), min(len(s.kd) - 1, f + 2)):
            for w in (9, 10):
                if s.kc[t, p, w] > 0.3 and s.kc[t - 1, p, w] > 0.3:
                    sp = max(sp, float(np.degrees(np.arccos(np.clip(s.kd[t, p, w] @ s.kd[t - 1, p, w], -1, 1)))))
        x += [sp, float(ok)]
        if s.wr is not None:                               # 3D wrists vs the shuttle's camera rays (tools/mv_wrist.py)
            if f not in s.wr_cache:
                s.wr_cache[f] = s.wr.reach_all(f)
            r = s.wr_cache[f]
            r = r[p] if p < len(r) else np.full(2, np.inf)
            mn = s.wr_cache[f].min(0)
            x += [min(r[0], 9.0), min(r[1], 9.0), min(r[0] - mn[0], 9.0) if np.isfinite(r[0]) else 9.0,
                  min(r[1] - mn[1], 9.0) if np.isfinite(r[1]) else 9.0]
        return x

    def snap(s, f):
        """refine a hit frame: the sound peak within +-2 frames if clear, else the fused peak"""
        for mode, thr in (("audio", 0.5), ("av", 0.0)):
            p = s.prob[mode]; a, b = max(0, (f - 2) * 8), min(len(p), (f + 3) * 8)
            if b > a and p[a:b].max() > thr:
                return int(round((a + int(np.argmax(p[a:b]))) / 8))
        return f


def frame_feats(r, f, peaks, n):
    """per-frame features: the candidate features without merging, distances to each cue's nearest peak, place in rally"""
    base = r.cand_feats(f, 0, 0, "")
    base = base[:10] + base[16:]                       # drop gaps and source flags (rebuilt below)
    for s in ("sound", "fused", "teacher", "turn"):
        pk = peaks.get(s, [])
        base.append(min([abs(f - q) for q in pk] or [30]))
    t0, t1 = (r.tfr[0], r.tfr[-1]) if len(r.tfr) else (0, n)
    base += [f - t0, t1 - f, f / max(n, 1)]
    if r.vid is not None and os.environ.get("HV_USE", "1") == "1":
        h = r.vid["hit"]; a, b = max(0, f - 1), min(len(h), f + 2)
        base += [float(h[f]) if f < len(h) else 0.0, float(h[a:b].max()) if b > a else 0.0,
                 float(r.vid["far"][min(f, len(h) - 1)])]
    return base


def dense(key_path, cand_path, src_path=None):
    """every frame scored (leave-one-rally-out), hits = peaks of the per-frame probability"""
    GT = json.load(open(key_path)); C = json.load(open(cand_path))
    M = models(); R = {}; X, Y, G, F = [], [], [], []
    for gi, (uid, hits) in enumerate(GT.items()):
        r = Rally(uid, M); R[uid] = r
        peaks = {}
        for c in C[uid]["cands"]:
            for s in c["src"]:
                peaks.setdefault(s, []).append(c["f"])
        n = len(r.foot); gf = np.array([h[0] for h in hits])
        for f in range(n):
            X.append(frame_feats(r, f, peaks, n)); Y.append(int(len(gf) > 0 and np.abs(gf - f).min() <= 1)); G.append(gi); F.append(f)
        print(uid, n, "frames", flush=True)
    X, Y, G, F = np.array(X, float), np.array(Y), np.array(G), np.array(F)
    prob = np.zeros(len(Y))
    for g in np.unique(G):
        tr = G != g
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=20, l2_regularization=1.0,
                                             class_weight={0: 1, 1: 3}).fit(X[tr], Y[tr])
        prob[G == g] = clf.predict_proba(X[G == g])[:, 1]
    np.savez(os.path.join(ROOT, "outputs", "cctv", os.environ.get("HP_EVAL", "hit_dense_eval.npz")), X=X, Y=Y, G=G, F=F, prob=prob)
    from scipy.signal import find_peaks
    uids = list(GT)
    print("threshold  found          extra  exact  within-1")
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
        tp = fp = fn = ex = w1 = 0
        for gi, uid in enumerate(uids):
            p = prob[G == gi]; p = np.convolve(p, [0.25, 0.5, 0.25], "same")
            pred = list(find_peaks(p, height=thr, distance=5)[0]); gf = [h[0] for h in GT[uid]]
            mm = match(pred, gf); tp += len(mm); fp += len(pred) - len(mm); fn += len(gf) - len(mm)
            ex += sum(pred[i] == gf[j] for i, j in mm); w1 += sum(abs(pred[i] - gf[j]) <= 1 for i, j in mm)
        n = tp + fn
        print(f"  {thr:.1f}     {tp}/{n} ({tp / n:.1%})  {fp:4d}   {ex / max(tp, 1):.0%}   {w1 / max(tp, 1):.0%}")


def match(pred, gt, tol=3):
    """one-to-one greedy matching by time: list of (pred index, gt index)"""
    pairs = sorted((abs(p - g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gt) if abs(p - g) <= tol)
    ui, uj, out = set(), set(), []
    for d, i, j in pairs:
        if i not in ui and j not in uj:
            ui.add(i); uj.add(j); out.append((i, j))
    return out


def main(key_path, cand_path):
    GT = json.load(open(key_path)); C = json.load(open(cand_path))
    M = models()
    R, X, Y, G = {}, [], [], []
    PX, PY, PG, PI = [], [], [], []
    for gi, (uid, hits) in enumerate(GT.items()):
        r = Rally(uid, M); R[uid] = r
        cf = [c["f"] for c in C[uid]["cands"]]; gf = [h[0] for h in hits]
        m = dict(match(cf, gf))
        for i, c in enumerate(C[uid]["cands"]):
            pg = cf[i] - cf[i - 1] if i else 99; ng = cf[i + 1] - cf[i] if i + 1 < len(cf) else 99
            X.append(r.cand_feats(c["f"], pg, ng, "".join(s_[0].upper() for s_ in c["src"]))); Y.append(int(i in m)); G.append(gi)
        for f, who in hits:
            p_true = int(who.split()[-1]) - 1
            for p in range(4):
                if np.isfinite(r.foot[min(f, len(r.foot) - 1), p]).all():
                    PX.append(r.player_feats(f, p)); PY.append(int(p == p_true)); PG.append(gi); PI.append((uid, f, p))
        print(uid, len(cf), "candidates", len(gf), "hits", flush=True)
    X, Y, G = np.array(X, float), np.array(Y), np.array(G)
    PX, PY, PG = np.array(PX, float), np.array(PY), np.array(PG)
    prob = np.zeros(len(Y)); pprob = np.zeros(len(PY))
    for g in np.unique(G):
        tr = G != g
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0).fit(X[tr], Y[tr])
        prob[G == g] = clf.predict_proba(X[G == g])[:, 1]
        trp = PG != g
        pc = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0).fit(PX[trp], PY[trp])
        pprob[PG == g] = pc.predict_proba(PX[PG == g])[:, 1]
    # evaluation per threshold: pick candidates above it, at least 5 frames apart (keep the surer), snap timing
    uids = list(GT)
    print("\nthreshold  found     extra  exact-frame  within-1  hitter(given true hit)")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        tp = fp = fn = 0; ex = w1 = 0
        k = 0
        for gi, uid in enumerate(uids):
            cands = C[uid]["cands"]; pr = prob[k:k + len(cands)]; k += len(cands)
            order = np.argsort(-pr); keep = []
            for i in order:
                if pr[i] >= thr and all(abs(cands[i]["f"] - cands[j]["f"]) >= 5 for j in keep):
                    keep.append(i)
            pred = sorted(R[uid].snap(cands[i]["f"]) for i in keep); gf = [h[0] for h in GT[uid]]
            mm = match(pred, gf); tp += len(mm); fp += len(pred) - len(mm); fn += len(gf) - len(mm)
            ex += sum(pred[i] == gf[j] for i, j in mm); w1 += sum(abs(pred[i] - gf[j]) <= 1 for i, j in mm)
        # hitter: among tracked players at each true hit, the highest score
        ok = tot = 0
        for key in {(u, f) for u, f, p in PI}:
            idx = [i for i, t in enumerate(PI) if (t[0], t[1]) == key]
            best = idx[int(np.argmax(pprob[idx]))]; tot += 1; ok += PY[best]
        n = tp + fn
        print(f"  {thr:.1f}     {tp}/{n} ({tp / n:.1%})  {fp:4d}   {ex / max(tp, 1):.0%}         {w1 / max(tp, 1):.0%}      {ok}/{tot} ({ok / tot:.0%})")
    np.savez(os.path.join(ROOT, "outputs", "cctv", "hit_pick_eval.npz"), X=X, Y=Y, G=G, prob=prob, PX=PX, PY=PY, PG=PG, pprob=pprob)


if __name__ == "__main__":
    (dense if os.environ.get("HP_DENSE") == "1" else main)(sys.argv[1], sys.argv[2])
