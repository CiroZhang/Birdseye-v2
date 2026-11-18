"""Data for the 3D rally viewer (docs/viewer): the final student on held-out CCTV rallies, nothing else.

Per rally: the broadcast camera (BirdsCourt), and per frame the student's shuttle, every present player's 24 SMPL
joints, each player's racket (grip at the racket-hand wrist, head 0.47 m along the predicted direction) and the hits
(peaks of the hit head). Display-only smoothing: joints, rackets and shuttle get a light centred moving average
(SMOOTH frames) so the page does not shimmer; nothing is moved towards any reference.
The side-view cross-check of the rally (results/recon_xview, cctv_t3.json) is copied in as its stats.

usage: ARCH=rallyphys STUDENT=1 BODY24=20 B24_SHAPE=5 FREE_S=1 TAG=t3 \
       python tools/export_viewer3d.py OUT_DIR uid [uid ...]
"""
import json, os, shutil, subprocess, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import train_rallynet as TR  # noqa: E402
from birdseye_v2 import rallynet as RN  # noqa: E402
from birdseye_v2.court import NET_Y  # noqa: E402
import imageio_ffmpeg  # noqa: E402

SMOOTH = int(os.environ.get("SMOOTH", "3"))
CONTACT = os.environ.get("CONTACT", "1") == "1"  # display-only racket-to-shuttle easing at hits
L_WRIST, R_WRIST = 20, 21  # SMPL-24


def smooth(x, ok, k=SMOOTH):
    """Centred moving average over time (axis 0) using only frames where ok (shape (T,)); frames not ok -> NaN."""
    T = len(x)
    out = np.full(x.shape, np.nan)
    for t in np.where(ok)[0]:
        a, b = max(0, t - k // 2), min(T, t + k // 2 + 1)
        m = ok[a:b]
        out[t] = x[a:b][m].mean(0)
    return out


def contact_swing24(S, J, u, pres, hits, hand, fps, sd=3.0, k=9, max_lean=0.35, max_gap=3.0):
    """Display only (as tools/export_rallynet_viewer.contact_swing, on SMPL-24): around each hit the hitter's racket
    head is eased onto the shuttle (Gaussian in time, sd frames) by re-solving shoulder -> elbow -> wrist with the
    arm's own bone lengths and elbow bend; a contact beyond reach leans the body towards it (<= max_lean m)."""
    from export_rallynet_viewer import _two_bone, _unit
    J, u = J.copy(), u.copy()
    T = len(S)
    Lr = RN.GRIP_TO_HEAD
    for h in hits:
        k0, q = int(h["f"]), int(h["slot"])
        if q < 0 or not np.isfinite(S[k0]).all() or not pres[k0, q]:
            continue
        sj, ej, wj, hj = (17, 19, 21, 23) if hand[q] else (16, 18, 20, 22)
        X = S[k0]
        if np.linalg.norm(J[k0, q, wj] + Lr * u[k0, q] - X) > max_gap:
            continue
        ua = np.linalg.norm(J[k0, q, ej] - J[k0, q, sj]); fa = np.linalg.norm(J[k0, q, wj] - J[k0, q, ej])
        short = max(np.linalg.norm(X - J[k0, q, sj]) - 0.97 * (ua + fa + Lr), 0.0)
        lean = min(short, max_lean) * _unit((X - J[k0, q, sj]) * np.array([1, 1, 0]))
        k2 = min(k0 + 3, T - 1)
        vout = (S[k2] - X) * fps / max(k2 - k0, 1) if np.isfinite(S[k2]).all() else np.zeros(3)
        vr = _unit(vout) * np.clip(0.4 * np.linalg.norm(vout), 4.0, 15.0) if np.linalg.norm(vout) > 1e-6 else np.zeros(3)
        for t in range(max(0, k0 - k), min(T, k0 + k + 1)):
            if not pres[t, q] or not np.isfinite(J[t, q]).all():
                continue
            g = float(np.exp(-0.5 * ((t - k0) / sd) ** 2))
            J[t, q] = J[t, q] + g * lean
            sh, el0, wr0 = J[t, q, sj], J[t, q, ej], J[t, q, wj]
            H = wr0 + Lr * u[t, q]
            H = H + g * (X + vr * (t - k0) / fps - H)
            dvec = _unit((1 - g) * u[t, q] + g * _unit(H - sh))
            el, wr = _two_bone(sh, el0, H - Lr * dvec, ua, fa)
            J[t, q, hj] = J[t, q, hj] + (wr - wr0)
            J[t, q, ej], J[t, q, wj] = el, wr
            u[t, q] = _unit(H - wr)
    return J, u


def steady_face(n, u, ok, k=5):
    """Display: a racket-face normal for every present frame, with no flips. Gaps are filled from the nearest frame
    that has one, the sign is kept consistent in time (a face has two sides), a centred k-frame average smooths
    single-frame flips, and the result is kept across the current shaft."""
    T = len(n)
    good = np.isfinite(n).all(1) & ok
    if not good.any():
        return n
    idx = np.where(good)[0]
    out = n.copy()
    for t in range(T):  # nearest frame with a normal
        if ok[t] and not good[t]:
            out[t] = n[idx[np.argmin(np.abs(idx - t))]]
    prev = None
    for t in range(T):  # consistent sign
        if ok[t] and np.isfinite(out[t]).all():
            if prev is not None and out[t] @ prev < 0:
                out[t] = -out[t]
            prev = out[t]
    sm = out.copy()
    for t in np.where(ok)[0]:
        a, b = max(0, t - k // 2), min(T, t + k // 2 + 1)
        m = ok[a:b] & np.isfinite(out[a:b]).all(1)
        v = out[a:b][m].mean(0)
        v = v - (v @ u[t]) * u[t] if np.isfinite(u[t]).all() else v
        sm[t] = v / (np.linalg.norm(v) + 1e-9)
    return np.where(ok[:, None], sm, np.nan)


def r(x, d=3):
    return None if x is None or not np.all(np.isfinite(x)) else np.round(np.asarray(x, float), d).tolist()


def main():
    out_dir, uids = sys.argv[1], sys.argv[2:]
    os.makedirs(os.path.join(out_dir, "d"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "v"), exist_ok=True)
    model = TR.load_model(os.environ.get("TAG", "t3"))
    xv = json.load(open(os.path.join(ROOT, "outputs", "xview", "cctv_t3.json")))
    index = []
    for uid in uids:
        f = TR.load_feats(os.path.join(TR.OBS, "cctv", uid + ".npz"))
        o = TR.infer(model, f)
        T = len(f["C"])
        fps = float(np.asarray(f["fps"]).reshape(-1)[0]) if "fps" in f else 25.0
        pres = np.asarray(f["present"]) > 0  # (T, P)
        J = o["joints24"]  # (T, P, 24, 3)
        P = J.shape[1]
        hand = (o["hand_p"].mean(0) > 0.5).astype(int)  # 1 = right hand holds the racket
        u = o["racket_u"]  # (T, P, 3)
        S = o["shuttle"]
        hp = 1 / (1 + np.exp(-o["hit"]))  # (T, P)
        pk = TR.hits_from(hp.max(1))
        Js = np.stack([smooth(J[:, q], pres[:, q]) for q in range(P)], 1)
        us = np.stack([smooth(u[:, q], pres[:, q]) for q in range(P)], 1)
        us = us / (np.linalg.norm(us, axis=-1, keepdims=True) + 1e-9)
        Ss = smooth(S, np.ones(T, bool), k=max(1, SMOOTH - 1))
        if CONTACT:
            hh = [{"f": int(k), "slot": int(np.argmax(hp[k]))} for k in pk]
            Js, us = contact_swing24(Ss, Js, us, pres, hh, hand, fps)
        players = []
        for q in range(P):
            if pres[:, q].sum() < 10:
                continue
            wr = Js[:, q, R_WRIST if hand[q] else L_WRIST]
            head = wr + RN.GRIP_TO_HEAD * us[:, q]
            from export_rallynet_viewer import face_normals
            fn = face_normals(np.where(pres[:, q, None], wr, np.nan), np.where(pres[:, q, None], us[:, q], np.nan))
            fn = steady_face(fn, us[:, q], pres[:, q])
            fr = np.where(pres[:, q])[0]
            players.append({
                "hand": "R" if hand[q] else "L",
                "f0": int(fr[0]), "f1": int(fr[-1]),
                "on": pres[:, q].astype(int).tolist(),
                "j": [r(Js[t, q].reshape(-1), 3) if pres[t, q] else None for t in range(T)],
                "rk": [r(np.r_[wr[t], head[t]], 3) if pres[t, q] else None for t in range(T)],
                "fn": [r(fn[t], 3) if pres[t, q] else None for t in range(T)],  # racket-face normal, turning smoothly
                "side": "near" if np.nanmedian(Js[pres[:, q], q, 0, 1]) < NET_Y else "far",
            })
        hits = [{"f": int(k), "p": int(np.argmax(hp[k]))} for k in pk]
        # who among the exported players: map slot -> exported index
        slot = [q for q in range(P) if pres[:, q].sum() >= 10]
        for h in hits:
            h["p"] = slot.index(h["p"]) if h["p"] in slot else -1
        cam = json.load(open(os.path.join(ROOT, "outputs", "cctv", "calib_bc", uid + ".json")))["cam"]
        st = xv.get(uid, {})
        d = {"uid": uid, "fps": fps, "T": T, "cam": cam, "shuttle": [r(Ss[t]) for t in range(T)],
             "players": players, "hits": hits,
             "stats": {"shuttle_cm": None if st.get("shuttle") is None else round(100 * st["shuttle"], 1),
                       "player_cm": None if st.get("player") is None else round(100 * st["player"], 1),
                       "n_shuttle": st.get("n_sh"), "n_player": st.get("n_pl")},
             "singles": len(players) <= 2, "display": {"smooth": SMOOTH, "contact": CONTACT}}
        json.dump(d, open(os.path.join(out_dir, "d", uid + ".json"), "w"), separators=(",", ":"))
        src = os.path.join(ROOT, "data", "cctv", "clips", uid + ".mp4")
        dst = os.path.join(out_dir, "v", uid + ".mp4")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-i", src, "-an", "-c:v", "libx264", "-preset", "slow",
                        "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", dst], check=True)
        index.append({"uid": uid, "sec": round(T / fps, 1), "players": len(players), "hits": len(hits), **d["stats"]})
        print(uid, T, "players", len(players), "hits", len(hits), d["stats"], flush=True)
    json.dump(index, open(os.path.join(out_dir, "d", "index.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
