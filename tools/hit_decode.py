"""Rally decoding for the hit picker: from every peak of the per-frame hit probability (low threshold), choose the
sequence of hits and hitters that best explains the evidence while the hitting side alternates across the net.

Inputs (leave-one-rally-out, as tools/hit_pick.py HP_DENSE=1 saved them): per-frame hit probability; per peak and
tracked player a hitter probability (model trained on the other rallies). Side of a player = side of the net his feet
are on at the peak. Score of a chosen hit = log-odds of the hit probability + log of its hitter's probability; a peak
left out scores 0; consecutive chosen hits must be on opposite sides and at least MIN_GAP frames apart.
usage: python tools/hit_decode.py <answer key json> <candidates json>"""
import json, os, sys

import numpy as np
from scipy.signal import find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import hit_pick as HP  # noqa: E402

MIN_GAP = 5
PRIOR = float(os.environ.get("HD_PRIOR", "0.2"))
SAME_SIDE = float(os.environ.get("HD_SAME", "-3.0"))   # log-penalty for two hits in a row on one side (soft rule)
NET_Y = HP.NET_Y


def decode(peaks, ph, players, pside=None):
    """peaks: frames; ph: hit probs; players: per peak list of (player, side, prob); pside: per peak P(near) from the
    side camera's shuttle or None. -> list of (frame, player)"""
    n = len(peaks)
    opts = []                       # per peak: best (score, player) for each side
    for i in range(n):
        lo = np.log(ph[i] / (1 - ph[i])) - np.log(PRIOR / (1 - PRIOR))   # include a peak when it beats PRIOR
        best = {}
        tot = {sd: sum(q for _, s2, q in players[i] if s2 == sd) for sd in ("near", "far")}
        sp = {}
        for side in ("near", "far"):
            if pside is not None and pside[i] is not None:
                sp[side] = pside[i] if side == "near" else 1 - pside[i]
            else:
                sp[side] = tot[side]
        top = max(sp.values()) if sp else 1.0
        for p, side, q in players[i]:
            # inclusion is decided by the hit evidence; the side costs only when it is not the likelier side
            sc = lo + np.log(max(sp[side], 1e-3) / max(top, 1e-3))
            if side not in best or q > best[side][2]:
                best[side] = (sc, p, q)
        best = {k: (v[0], v[1]) for k, v in best.items()}
        opts.append(best)
    # DP over (peak, side): best total ending with a hit at this peak on this side
    D = {}; back = {}
    for i in range(n):
        for side, (sc, p) in opts[i].items():
            best, arg = sc, None
            for j in range(i):
                if peaks[i] - peaks[j] < MIN_GAP:
                    continue
                for s2 in opts[j]:
                    v = D.get((j, s2), -1e9) + sc + (SAME_SIDE if s2 == side else 0.0)
                    if v > best:
                        best, arg = v, (j, s2)
            D[(i, side)] = best; back[(i, side)] = arg
    if not D:
        return []
    end = max(D, key=D.get)
    if D[end] <= 0:
        return []
    out = []
    while end is not None:
        i, side = end; out.append((peaks[i], opts[i][side][1])); end = back[end]
    return out[::-1]


def main(key_path, cand_path):
    GT = json.load(open(key_path)); C = json.load(open(cand_path)); uids = list(GT)
    z = np.load(os.path.join(ROOT, "outputs", "cctv", os.environ.get("HP_EVAL", "hit_dense_eval.npz"))); prob, G = z["prob"], z["G"]
    M = HP.models(); R = {u: HP.Rally(u, M) for u in uids}
    import body_mv as BM, cctv_tri as CT
    SIDE = {}
    for u in uids:
        sd, sh = CT.side_camera(u); cam = BM.static_cam(sd)
        rr = json.load(open(os.path.join(ROOT, "outputs", "cctv_side", "refined", u + ".json")))
        tr = {int(f) - sh: np.array(p, float) for f, p in zip(rr["frames"], rr["px"]) if p is not None}
        netx = BM.project(cam, np.array([[3.05, NET_Y, z] for z in (0.5, 1.0, 1.5, 2.0)]))[0][:, 0].mean()
        nearx = BM.project(cam, np.array([[3.05, 2.0, 1.0]]))[0][0, 0]
        SIDE[u] = (tr, netx, nearx)
    # hitter training rows at true hits (all rallies; used leave-one-rally-out)
    PX, PY, PG = [], [], []
    for gi, u in enumerate(uids):
        r = R[u]
        for f, who in GT[u]:
            pt = int(who.split()[-1]) - 1
            for p in range(4):
                if np.isfinite(r.foot[min(f, len(r.foot) - 1), p]).all():
                    PX.append(r.player_feats(f, p)); PY.append(int(p == pt)); PG.append(gi)
    PX, PY, PG = np.array(PX, float), np.array(PY), np.array(PG)
    for thr_peak in (0.05,):
        tp = fp = fn = ex = w1 = hit_ok = side_ok = tracked = hit_ok_t = 0
        per = {}                    # per rally: found, extra, missed, hitter right (HD_DUMP=<json> saves it)
        hits_out = {}               # per rally: decoded [frame, hitter slot] (HD_HITS=<json> saves it)
        tp_raw = fp_raw = 0
        for gi, u in enumerate(uids):
            r = R[u]
            pc = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0).fit(PX[PG != gi], PY[PG != gi])
            if os.environ.get("HD_PROB") == "video":          # the video model's own per-frame probability
                p = np.convolve(r.vid["hit"], [0.25, 0.5, 0.25], "same")
            elif os.environ.get("HD_PROB", "").startswith("mix"):   # mix<w>: w * video + (1 - w) * picker
                w = float(os.environ["HD_PROB"][3:] or 0.5); q = prob[G == gi]; v = r.vid["hit"][:len(q)]
                p = np.convolve(w * v + (1 - w) * q, [0.25, 0.5, 0.25], "same")
            else:                                              # the picker's (leave-one-rally-out) probability
                p = np.convolve(prob[G == gi], [0.25, 0.5, 0.25], "same")
            pk = [int(i) - 1 for i in find_peaks(np.r_[0.0, p, 0.0], height=thr_peak, distance=3)[0]]; ph = np.clip(p[pk], 1e-3, 1 - 1e-3)  # padded: a hit on the first or last frame is a peak too
            players = []
            for f in pk:
                rows, ids = [], []
                for q in range(4):
                    ft = r.foot[min(f, len(r.foot) - 1), q]
                    if np.isfinite(ft).all():
                        rows.append(r.player_feats(f, q)); ids.append((q, "near" if ft[1] < NET_Y else "far"))
                pq = pc.predict_proba(np.array(rows, float))[:, 1] if rows else []
                pq = pq / max(pq.sum(), 1e-9) if len(rows) else pq
                players.append([(q, sd, float(w)) for (q, sd), w in zip(ids, pq)])
            tr, netx, nearx = SIDE[u]; ps = []
            for f in pk:
                ks = [k for k in range(f - 3, f + 4) if k in tr]
                mode = os.environ.get("HV_SIDE", "combo")
                vid_ok = r.vid is not None and mode in ("1", "combo")
                if vid_ok and (mode == "1" or not ks):
                    ps.append(float(np.clip(1 - r.vid["far"][min(f, len(r.vid["far"]) - 1)], 0.05, 0.95)))   # video model's side
                elif ks:
                    x = tr[min(ks, key=lambda k: abs(k - f))][0]; ps.append(0.95 if (x < netx) == (nearx < netx) else 0.05)
                else:
                    ps.append(None)
            dec = decode(pk, ph, players, ps)
            pred = [f for f, _ in dec]; gf = [h[0] for h in GT[u]]
            mm = HP.match(pred, gf); tp += len(mm); fp += len(pred) - len(mm); fn += len(gf) - len(mm)
            ex += sum(pred[i] == gf[j] for i, j in mm); w1 += sum(abs(pred[i] - gf[j]) <= 1 for i, j in mm)
            per[u] = [len(mm), len(pred) - len(mm), len(gf) - len(mm), 0]
            hits_out[u] = [[int(f_), int(p_)] for f_, p_ in dec]
            for i, j in mm:
                pt = int(GT[u][j][1].split()[-1]) - 1
                hit_ok += dec[i][1] == pt; per[u][3] += int(dec[i][1] == pt)
                ft = r.foot[min(gf[j], len(r.foot) - 1)]
                if np.isfinite(ft[pt]).all():
                    tracked += 1
                    side_ok += np.isfinite(ft[dec[i][1]]).all() and (ft[pt][1] < NET_Y) == (ft[dec[i][1]][1] < NET_Y)
                    hit_ok_t += dec[i][1] == pt
            raw = [f for f, q in zip(pk, ph) if q >= 0.2]
            m2 = HP.match(raw, gf); tp_raw += len(m2); fp_raw += len(raw) - len(m2)
            bad = [gf[j] for j in range(len(gf)) if j not in {b for a, b in mm}]
            extra = [pred[i] for i in range(len(pred)) if i not in {a for a, b in mm}]
            if bad or extra:
                print(f"  {u}: missed {bad}  extra {extra}")
        if os.environ.get("HD_HITS"):
            json.dump(hits_out, open(os.environ["HD_HITS"], "w"))
        if os.environ.get("HD_DUMP"):
            json.dump(per, open(os.environ["HD_DUMP"], "w"))
        n = tp + fn
        print(f"\nwithout the rally rule (peaks >= 0.2): found {tp_raw}/{n} ({tp_raw / n:.1%}), extra {fp_raw}")
        print(f"with the rally rule:                  found {tp}/{n} ({tp / n:.1%}), extra {fp}; timing exact {ex / max(tp, 1):.0%}, "
              f"within 1 frame {w1 / max(tp, 1):.0%}; hitter right {hit_ok}/{tp} ({hit_ok / max(tp, 1):.0%}); where the true hitter is tracked: hitter {hit_ok_t}/{tracked} ({hit_ok_t / max(tracked, 1):.0%}), side {side_ok}/{tracked} ({side_ok / max(tracked, 1):.0%})")


if __name__ == "__main__" and sys.argv[1] != "--label":
    main(sys.argv[1], sys.argv[2])


def side_info(u):
    """the side camera's shuttle track (TV frames) and where the net and the near side are in its image"""
    import body_mv as BM, cctv_tri as CT
    sd, sh = CT.side_camera(u); cam = BM.static_cam(sd)
    rr = json.load(open(os.path.join(ROOT, "outputs", "cctv_side", "refined", u + ".json")))
    tr = {int(f) - sh: np.array(p, float) for f, p in zip(rr["frames"], rr["px"]) if p is not None}
    netx = BM.project(cam, np.array([[3.05, NET_Y, z] for z in (0.5, 1.0, 1.5, 2.0)]))[0][:, 0].mean()
    nearx = BM.project(cam, np.array([[3.05, 2.0, 1.0]]))[0][0, 0]
    return tr, netx, nearx


def label_all(key_path, uids, out_path):
    """Teacher hits for any rallies: peaks of the video model's per-frame hit probability (HV_DIR), hitters from a
    hitter model trained on every answer-key rally, decoded with the side rule (HD_PRIOR, HD_SAME). Writes
    {uid: [[frame, track, hit prob, hitter prob], ...]} (tracks as in tools/mv_track.py with HP_TRACK=mv)."""
    GT = json.load(open(key_path)); M = HP.models()
    PX, PY = [], []
    for u in GT:
        r = HP.Rally(u, M)
        for f, who in GT[u]:
            pt = int(who.split()[-1]) - 1
            for p in range(4):
                if np.isfinite(r.foot[min(f, len(r.foot) - 1), p]).all():
                    PX.append(r.player_feats(f, p)); PY.append(int(p == pt))
    pc = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0).fit(np.array(PX, float), np.array(PY))
    out = json.load(open(out_path)) if os.path.exists(out_path) else {}
    for n_, u in enumerate(uids):
        if u in out:
            continue
        try:
            r = HP.Rally(u, M)
            p = np.convolve(r.vid["hit"], [0.25, 0.5, 0.25], "same")
            pk = [int(i) - 1 for i in find_peaks(np.r_[0.0, p, 0.0], height=0.05, distance=3)[0]]; ph = np.clip(p[pk], 1e-3, 1 - 1e-3)
            players = []
            for f in pk:
                rows, ids = [], []
                for q in range(4):
                    ft = r.foot[min(f, len(r.foot) - 1), q]
                    if np.isfinite(ft).all():
                        rows.append(r.player_feats(f, q)); ids.append((q, "near" if ft[1] < NET_Y else "far"))
                pq = pc.predict_proba(np.array(rows, float))[:, 1] if rows else []
                pq = pq / max(pq.sum(), 1e-9) if len(rows) else pq
                players.append([(q, sd, float(w)) for (q, sd), w in zip(ids, pq)])
            tr, netx, nearx = side_info(u); ps = []
            for f in pk:
                ks = [k for k in range(f - 3, f + 4) if k in tr]
                if not ks:
                    ps.append(float(np.clip(1 - r.vid["far"][min(f, len(r.vid["far"]) - 1)], 0.05, 0.95)))
                else:
                    x = tr[min(ks, key=lambda k: abs(k - f))][0]; ps.append(0.95 if (x < netx) == (nearx < netx) else 0.05)
            dec = decode(pk, ph, players, ps)
            hp = dict(zip(pk, ph)); pp = {f: {q: w for q, _, w in pl} for f, pl in zip(pk, players)}
            out[u] = [[int(f), int(q), round(float(hp[f]), 3), round(float(pp[f].get(q, 0.0)), 3)] for f, q in dec]
        except Exception as e:
            print(u, "FAILED", repr(e), flush=True)
        if n_ % 25 == 0:
            json.dump(out, open(out_path, "w")); print(n_, len(uids), u, len(out.get(u, [])), "hits", flush=True)
    json.dump(out, open(out_path, "w"))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--label":
    # usage: python tools/hit_decode.py --label <answer key> <out json> uid ...
    label_all(sys.argv[2], sys.argv[4:], sys.argv[3])
