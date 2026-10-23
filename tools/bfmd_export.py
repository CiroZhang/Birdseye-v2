"""Our BFMD reconstruction in Birdseye v1's layout, so v1's own builder/trainer
(third_party/birdseye_v1/action/build_training_data_selfdetected.py, train_final.py) runs on it unchanged.

  in   outputs/bfmd/<VIMO_DIR>/<match>__g<G>_r<RRR>.npz   (tools/fb_vimo.py, FB_SET=bfmd)
  out  data/bfmd/<OUT_SD>/<match>/joints_selfdetected.json  {"rallies": {"g1_r000": {start, end, joints: {top, bottom}}}}

BFMD is 25 fps shown at 30 (one repeated frame in six); fb_vimo keeps the native frames, so each 30 fps frame
takes the body of the native frame it repeats -- the hit labels are 30 fps frame numbers.
usage: python tools/bfmd_export.py   (env VIMO_DIR=vimo_bc, OUT_SD=v2_sd)"""
import json, os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "bfmd")
VIMO = os.path.join(ROOT, "outputs", "bfmd", os.environ.get("VIMO_DIR", "vimo_bc"))
OUT_SD = os.path.join(DATA, os.environ.get("OUT_SD", "v2_sd"))


def to_30fps(J, frames, n):
    """J: (T_native,24,3) at the kept 30 fps indices `frames` -> (n,24,3), each frame from the last kept one."""
    src = np.searchsorted(frames, np.arange(n), side="right") - 1
    out = np.full((n, 24, 3), np.nan)
    ok = (src >= 0) & (src < len(J))
    out[ok] = J[src[ok]]
    return out


def main():
    matches = open(os.path.join(DATA, "meta", "all_matches.txt")).read().split()
    for m in matches:
        rallies = json.load(open(os.path.join(DATA, "meta", m + ".rallies.json")))
        out, cover = {}, []
        for r in rallies:
            tag = f"g{r['game']}_r{r['rally']:03d}"
            p = os.path.join(VIMO, f"{m}__{tag}.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            n = r["end"] - r["start"] + 1
            joints = {}
            for s in ("top", "bottom"):
                J = to_30fps(d[s], d["frames"], n)
                if np.isfinite(J[:, 0, 0]).sum() >= 8:  # v1 skips a side with < 8 boxes
                    joints[s] = np.round(J, 4).tolist()
                    cover.append(np.isfinite(J[:, 0, 0]).mean())
            out[tag] = {"start": r["start"], "end": r["end"], "joints": joints}
        os.makedirs(os.path.join(OUT_SD, m), exist_ok=True)
        json.dump({"rallies": out}, open(os.path.join(OUT_SD, m, "joints_selfdetected.json"), "w"))
        print(f"{m}: {len(out)}/{len(rallies)} rallies, side coverage {np.mean(cover) if cover else 0:.3f}", flush=True)


if __name__ == "__main__":
    main()
