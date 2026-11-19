"""Teacher mode of the 3D rally viewer (docs/viewer): the multi-view teacher's 3D (bodies, rackets, shuttle with its
labelled / unsure flag, hits) for the training-match rallies of the earlier teacher viewer (docs/teacher/d/*.json),
converted to the viewer's format, with the final student's output on the same frames. The 2x2 four-camera video of
each rally starts at the rally's first frame (k0). Run with the student's env (see tools/export_viewer3d.py).

usage: python tools/export_viewer3d_teacher.py SRC_DIR(docs/teacher) OUT_DIR(docs/viewer) uid [uid ...]
"""
import json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from export_viewer3d import steady_face, r, student_rally, merge_index, merge_side  # noqa: E402
import train_rallynet as TR  # noqa: E402
from export_rallynet_viewer import face_normals  # noqa: E402

GRIP_TO_HEAD = 0.47


def main():
    src, out, uids = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(os.path.join(out, "dt"), exist_ok=True)
    idx_path = os.path.join(out, "dt", "index.json")
    index = []
    model = TR.load_model(os.environ.get("TAG", "t3"))
    for uid in uids:
        d = json.load(open(os.path.join(src, "d", uid + ".json")))
        k0, k1, fps = int(d["k0"]), int(d["k1"]), float(d.get("fps", 25))
        T = k1 - k0 + 1
        sh = [None] * T
        lab = [0] * T
        for row in d["path"]:
            t = int(row[0]) - k0
            if 0 <= t < T:
                sh[t] = r(row[1:4])
                lab[t] = int(row[4])
        players = []
        for p in d["players"]:
            J = np.full((T, 24, 3), np.nan)
            for f, j in p["frames"].items():
                t = int(f) - k0
                if 0 <= t < T and j is not None and len(j) == 24:
                    J[t] = np.asarray(j, float)
            on = np.isfinite(J).all((1, 2))
            if on.sum() < 10:
                continue
            g = np.full((T, 3), np.nan)
            u = np.full((T, 3), np.nan)
            for f, v in (p.get("rkfit") or {}).items():
                t = int(f) - k0
                if 0 <= t < T:
                    g[t], u[t] = v[0:3], v[3:6]
            rk_on = on & np.isfinite(u).all(1)
            u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
            fn = steady_face(face_normals(np.where(rk_on[:, None], g, np.nan), np.where(rk_on[:, None], u, np.nan)), u, rk_on)
            head = g + GRIP_TO_HEAD * u
            side = "near" if np.nanmedian(J[on, 0, 1]) < 6.7 else "far"
            players.append({"hand": "R", "side": side, "f0": int(np.where(on)[0][0]), "f1": int(np.where(on)[0][-1]),
                            "on": on.astype(int).tolist(),
                            "j": [r(J[t].reshape(-1)) if on[t] else None for t in range(T)],
                            "rk": [r(np.r_[g[t], head[t]]) if rk_on[t] else None for t in range(T)],
                            "fn": [r(fn[t]) if rk_on[t] and np.isfinite(fn[t]).all() else None for t in range(T)]})
        hits = [{"f": int(h) - k0, "p": -1} for h in d["hits"] if 0 <= int(h) - k0 < T]
        st = d.get("stats", {})
        o = {"uid": uid, "mode": "teacher", "fps": fps, "T": T, "shuttle": sh, "label": lab, "players": players,
             "hits": hits, "video": f"../teacher/v/{uid}.mp4", "cams": d.get("cams"),
             "stats": {"labelled_frac": round(100 * float(d.get("conf_frac", 0)), 0), "hits": len(hits),
                       "student_cm": st.get("med_cm")},
             "singles": len(players) <= 2,
             "student": merge_side(student_rally(model, uid, k0, T), 1 if len(players) <= 2 else 2)}  # the final student on the same frames, for comparison
        json.dump(o, open(os.path.join(out, "dt", uid + ".json"), "w"), separators=(",", ":"))
        index.append({"uid": uid, "mode": "teacher", "sec": round(T / fps, 1), "players": len(players), "hits": len(hits),
                      "labelled": o["stats"]["labelled_frac"]})
        print(uid, T, "players", len(players), "hits", len(hits), flush=True)
    merge_index(idx_path, index)


if __name__ == "__main__":
    main()
