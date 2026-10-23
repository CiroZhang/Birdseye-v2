"""Download DroneData study videos (public Drive of Ding et al.) for given matches.
Per rally: 1.mp4 = top view, (1).mp4 = back view. Skips files already present.
usage: python tools/download_videos.py BD12 BD13 ..."""
import json, os, sys, time, requests, gdown
from gdown.download_folder import _parse_embedded_folder_view

STUDY_ROOT = "1sIKIDLjyhccO_y6gXeaIkr_1gu1o0vYw"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "videos")
sess = requests.Session()

def children(fid):
    for _ in range(4):
        try:
            return _parse_embedded_folder_view(sess=sess, folder_id=fid, verify=True)[1]
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"cannot list {fid}")

root = {n: i for i, n, _ in children(STUDY_ROOT)}
matches = {n: i for i, n, _ in children(root["Study_Videos"])}
for m in sys.argv[1:]:
    for rid, rname, _ in children(matches[m]):
        d = os.path.join(OUT, m, rname); os.makedirs(d, exist_ok=True)
        for vid, vname, _ in children(rid):
            dst = os.path.join(d, vname)
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                continue
            for _ in range(3):
                try:
                    gdown.download(id=vid, output=dst, quiet=True); break
                except Exception as e:
                    print("retry", dst, e); time.sleep(5)
        print(m, rname, sorted(os.listdir(d)), flush=True)
