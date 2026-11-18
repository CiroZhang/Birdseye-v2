"""2D-keypoint baseline for the BFMD stroke classifier, in Birdseye v1's layout (as tools/bfmd_export.py does for
VIMO), so the same classifier (tools/bfmd_stroke.py) sees the same rallies with 2D instead of 3D bodies.

Per side, the player is the YOLOv8-pose detection (outputs/bfmd/pose1280) whose mid-ankle is nearest the ankle pixel
that tools/fb_vimo.py tracked for that side (outputs/bfmd/vimo_bc/<uid>.npz *_ankle_px), so 2D and 3D follow the same
person. Image coordinates are divided by the image height and written as (x, y, z) = (u/H, 0, -v/H): z is up, so
v1's canonicalisation (pelvis-relative; the far player's x mirrored) applies unchanged. Joints with confidence
< MIN_CONF are missing. COCO-17 goes into the SMPL-24 slots as in tools/rallynet_bfmd_export.py.

out  data/bfmd/<OUT_SD>/<match>/joints_selfdetected.json
usage: python tools/bfmd_export_2d.py   (env OUT_SD=k2d_sd)"""
import json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from birdseye_v2.players import load_pose  # noqa: E402
from bfmd_export import to_30fps  # noqa: E402
from rallynet_bfmd_export import smpl24  # noqa: E402

DATA = os.path.join(ROOT, "data", "bfmd")
OUT = os.path.join(ROOT, "outputs", "bfmd")
OUT_SD = os.path.join(DATA, os.environ.get("OUT_SD", "k2d_sd"))
MIN_CONF = 0.2
MAX_PX = 80.0  # a detection's mid-ankle must lie this close (px) to the tracked ankle


def mid_ankle(kp):
    a = kp[:, [15, 16]]
    ok = a[..., 2] > 0.3
    m = np.where(ok.any(1)[:, None], (a[..., :2] * ok[..., None]).sum(1) / np.maximum(ok.sum(1), 1)[:, None], np.nan)
    return m


def side_kp(pose, frames, ankle_px, H):
    """(T_native,17,3) normalised 2D keypoints of the tracked player, NaN where not found."""
    out = np.full((len(frames), 17, 3), np.nan)
    for i, f in enumerate(frames):
        if f >= len(pose) or not len(pose[f]) or not np.isfinite(ankle_px[i]).all():
            continue
        kp = pose[f]
        d = np.linalg.norm(mid_ankle(kp) - ankle_px[i], axis=1)
        if not np.isfinite(d).any() or np.nanmin(d) > MAX_PX:
            continue
        k = kp[int(np.nanargmin(d))]
        xyz = np.c_[k[:, 0] / H, np.zeros(17), -k[:, 1] / H]
        xyz[k[:, 2] < MIN_CONF] = np.nan
        out[i] = xyz
    return out


def main():
    matches = open(os.path.join(DATA, "meta", "all_matches.txt")).read().split()
    for m in matches:
        rallies = json.load(open(os.path.join(DATA, "meta", m + ".rallies.json")))
        out, cover = {}, []
        for r in rallies:
            tag = f"g{r['game']}_r{r['rally']:03d}"
            uid = f"{m}__{tag}"
            vp, pp = os.path.join(OUT, "vimo_bc", uid + ".npz"), os.path.join(OUT, "pose1280", uid + ".npz")
            cp = os.path.join(OUT, "calib_bc", uid + ".json")
            if not (os.path.exists(vp) and os.path.exists(pp)):
                continue
            d, pose = np.load(vp), load_pose(pp)
            cam = json.load(open(cp)).get("cam") if os.path.exists(cp) else None
            H = cam["size"][1] if cam else 720  # clips are 1280x720 when uncalibrated
            n = r["end"] - r["start"] + 1
            joints = {}
            for s in ("top", "bottom"):
                if s + "_ankle_px" not in d.files:
                    continue
                K = side_kp(pose, d["frames"], d[s + "_ankle_px"], H)
                with np.errstate(invalid="ignore"):
                    J = to_30fps(smpl24(K), d["frames"], n)
                if np.isfinite(J[:, 0, 0]).sum() >= 8:
                    joints[s] = np.round(J, 4).tolist()
                    cover.append(np.isfinite(J[:, 0, 0]).mean())
            out[tag] = {"start": r["start"], "end": r["end"], "joints": joints}
        os.makedirs(os.path.join(OUT_SD, m), exist_ok=True)
        json.dump({"rallies": out}, open(os.path.join(OUT_SD, m, "joints_selfdetected.json"), "w"))
        print(f"{m}: {len(out)}/{len(rallies)} rallies, side coverage {np.mean(cover) if cover else 0:.3f}", flush=True)


if __name__ == "__main__":
    main()
