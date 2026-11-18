"""Shrink GVHMR outputs (run_gvhmr_2p.py npz) to what the cross-view scorer needs: per player the frame indices and the
court-frame SMPL ankles (joints 7, 8). Usage: python gvhmr_extract.py IN_DIR OUT_DIR"""
import sys, os, glob
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for p in sorted(glob.glob(os.path.join(src, "*.npz"))):
    v = np.load(p, allow_pickle=True)
    n = int(v["n_players"])
    out = dict(n_players=n, n_frames=int(v["n_frames"]), timing=v["timing"], gpu=v["gpu"])
    for i in range(n):
        out[f"p{i}_frames"] = v[f"p{i}_frames"]
        out[f"p{i}_ankles"] = v[f"p{i}_joints_court"][:, [7, 8]]
    np.savez_compressed(os.path.join(dst, os.path.basename(p)), **out)
print(len(glob.glob(os.path.join(dst, "*.npz"))))
