"""FineBadminton-20K stroke classification in Birdseye v1's layout, so tools/bfmd_stroke.py (v1's classifier, clean
protocol, match-level split) runs on it exactly as on BFMD.

One "match" = one FineBadminton source match (48); rally k -> tag g1_r<k>. Hits come from the FineBadminton-20K
annotations (data/fb20k/clips/<match>_<k>.json: frame, hitter top/bottom, stroke type), clip-local source frames.
Bodies, per representation REP:
  vimo   VIMO world joints (outputs/fb20k/vimo_bc), as tools/bfmd_export.py
  2d     YOLOv8-pose keypoints of the same tracked players (outputs/fb20k/pose1280), as tools/bfmd_export_2d.py
  <tag>  a RallyNet/RallyPhys student's SMPL-24 joints (outputs/rally_obs/fb20k), as tools/rallynet_bfmd_export.py
All are mapped to source frames through the kept-frame list of the VIMO run (repeated frames take the last kept one).

out  data/fbs/<REP>_sd/<match>/joints_selfdetected.json, data/fbs/<REP>/annotations/hit_inferred/<match>.json
then BFMD_DATA_ROOT=data/fbs/<REP> BFMD_SELFDETECTED_ROOT=data/fbs/<REP>_sd python third_party/birdseye_v1/action/build_training_data_selfdetected.py
usage: REP=vimo|2d|t3 python tools/fbs_export.py   (student: the model's env, as for evaluation)"""
import glob, json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from bfmd_export import to_30fps  # noqa: E402
from bfmd_export_2d import side_kp  # noqa: E402
from rallynet_bfmd_export import smpl24  # noqa: E402

REP = os.environ.get("REP", "vimo")
OWN = os.environ.get("OWN", "0") == "1"  # student REP: also its shuttle, racket_<side> and hit_<side> (FEAT=own)
FILL = os.environ.get("FILL", "0") == "1"
OUT_REP = os.environ.get("OUT_REP", REP + ("own" if OWN else ""))
CLIPS = os.path.join(ROOT, "data", "fb20k", "clips")
OUTS = os.path.join(ROOT, "outputs", "fb20k")
DST = os.path.join(ROOT, "data", "fbs")


def student_joints(model, TR, RN, uid, frames):
    """{side: (n_native,24,3)} from the student, the most-present slot on each side of the net (gaps from others).
    The student runs on every video frame (rally_obs), so native frame i is its raw frame frames[i] (FineBadminton
    clips repeat every 6th frame: indexing by i drifts by up to seconds)."""
    n_native = len(frames)
    frames = np.asarray(frames, int)

    def at_native(A, fill=np.nan):
        B = np.full((n_native,) + A.shape[1:], fill)
        ok = frames < len(A)
        B[ok] = A[frames[ok]]
        return B
    from birdseye_v2.court import NET_Y
    p = os.path.join(TR.OBS, "fb20k", uid + ".npz")
    if not os.path.exists(p):
        return {}
    f = TR.load_feats(p)
    res = TR.infer(model, f)
    J24, pres = res["joints24"], f["present"] > 0
    out = {}
    if OWN:  # the student's shuttle, racket head per slot (as rallynet_bfmd_export.py) and hit probability
        hand = (np.nanmean(np.where(pres, res["hand_p"], np.nan), 0) > 0.5).astype(int)
        u = res["u2"][:, np.arange(pres.shape[1]), hand]
        u = u / (np.linalg.norm(u, axis=-1, keepdims=True) + 1e-9)
        rk = res["w2"][:, np.arange(pres.shape[1]), hand] + 0.38 * u
        hp = (1 / (1 + np.exp(-res["hit"]))) * pres
        out["shuttle"] = at_native(res["shuttle"])
    for side in ("top", "bottom"):
        slots = [q for q in range(RN.P) if pres[:, q].sum() >= 8
                 and (np.median(f["root"][pres[:, q], q, 1]) > NET_Y) == (side == "top")]
        if not slots:
            continue
        slots.sort(key=lambda q: -pres[:, q].sum())
        K = np.full(J24.shape[:1] + (24, 3), np.nan)
        for q in slots:
            gap = ~np.isfinite(K[:, 0, 0]) & pres[:, q]
            K[gap] = J24[gap, q]
        if FILL:  # as rallynet_bfmd_export.py FILL=1: the most-present slot's prediction through gaps inside the track
            have = np.flatnonzero(np.isfinite(K[:, 0, 0]))
            gfill = np.zeros(len(K), bool)
            if len(have):
                gfill[have[0]:have[-1] + 1] = ~np.isfinite(K[have[0]:have[-1] + 1, 0, 0])
                K[gfill] = J24[gfill, slots[0]]
        out[side] = at_native(K)
        if OWN:
            R = np.full(K.shape[:1] + (3,), np.nan)
            for q in slots:
                gap = ~np.isfinite(R[:, 0]) & pres[:, q]
                R[gap] = rk[gap, q]
            if FILL:
                R[gfill] = rk[gfill, slots[0]]
            out["racket_" + side] = at_native(R)
            out["hit_" + side] = at_native(hp[:, slots].max(1), 0.0)
    return out


def main():
    model = TR = RN = None
    if REP not in ("vimo", "2d"):
        import train_rallynet as TR
        from birdseye_v2 import rallynet as RN
        model = TR.load_model(REP)
    from birdseye_v2.players import load_pose
    by_match = {}
    for p in sorted(glob.glob(os.path.join(CLIPS, "*.json"))):
        uid = os.path.basename(p)[:-5]
        by_match.setdefault(uid.rsplit("_", 1)[0], []).append(uid)
    for m, uids in sorted(by_match.items()):
        rallies, hits, cover = {}, [], []
        for uid in uids:
            vp = os.path.join(OUTS, "vimo_bc", uid + ".npz")
            if not os.path.exists(vp):
                continue
            d = np.load(vp)
            frames = d["frames"]
            lab = json.load(open(os.path.join(CLIPS, uid + ".json"), encoding="utf8"))
            n = int(max(frames.max() + 1, max([h["frame"] for h in lab["hits"]], default=0) + 1))
            k = int(uid.rsplit("_", 1)[1])
            tag = f"g1_r{k:03d}"
            if REP == "vimo":
                src = {s: d[s] for s in ("top", "bottom")}
            elif REP == "2d":
                pp, cp = os.path.join(OUTS, "pose1280", uid + ".npz"), os.path.join(OUTS, "calib_bc", uid + ".json")
                if not os.path.exists(pp):
                    continue
                pose = load_pose(pp)
                cam = json.load(open(cp)).get("cam") if os.path.exists(cp) else None
                H = cam["size"][1] if cam else 720  # clips are 1280x720 when uncalibrated
                src = {}
                for s in ("top", "bottom"):
                    if s + "_ankle_px" in d.files:
                        with np.errstate(invalid="ignore"):
                            src[s] = smpl24(side_kp(pose, frames, d[s + "_ankle_px"], H))
            else:
                src = student_joints(model, TR, RN, uid, frames)
            extra = {}
            for key in [k_ for k_ in src if k_ not in ("top", "bottom")]:  # OWN: per-frame extras, as the joints
                A = src.pop(key)
                at = np.searchsorted(frames, np.arange(n), side="right") - 1
                ok = (at >= 0) & (at < len(A))
                B = np.full((n,) + A.shape[1:], np.nan if key[:3] != "hit" else 0.0)
                B[ok] = A[at[ok]]
                extra[key] = np.round(B, 4).tolist()
            joints = {}
            for s, J in src.items():
                J = to_30fps(np.asarray(J, float), frames, n)
                if np.isfinite(J[:, 0, 0]).sum() >= 8:
                    joints[s] = np.round(J, 4).tolist()
                    cover.append(np.isfinite(J[:, 0, 0]).mean())
            rallies[tag] = {"start": 0, "end": n - 1, "joints": joints, **extra}
            for h in lab["hits"]:
                if h.get("type") and h.get("hitter") in ("top", "bottom"):
                    hits.append({"game": 1, "rally": k, "frame": int(h["frame"]), "side": h["hitter"], "shot_type": h["type"]})
        os.makedirs(os.path.join(DST, OUT_REP + "_sd", m), exist_ok=True)
        os.makedirs(os.path.join(DST, OUT_REP, "annotations", "hit_inferred"), exist_ok=True)
        json.dump({"rallies": rallies}, open(os.path.join(DST, OUT_REP + "_sd", m, "joints_selfdetected.json"), "w"))
        json.dump({"hits": hits}, open(os.path.join(DST, OUT_REP, "annotations", "hit_inferred", m + ".json"), "w"))
        print(f"{m}: {len(rallies)} rallies, {len(hits)} hits, side coverage {np.mean(cover) if cover else 0:.3f}", flush=True)


if __name__ == "__main__":
    main()
