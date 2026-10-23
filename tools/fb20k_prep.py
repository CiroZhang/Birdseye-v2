"""Cut every annotated rally of the FineBadminton-20K training matches into its own native-25fps clip
and write its labels in our format. Matches listed in benchmark_source_videos.txt (used to build
FBBench) are never touched, so nothing here can leak into the benchmark.

Output: data/fb20k/clips/<match>_<k>.mp4   rally [start-PAD, end+PAD], resized to 1280x720, source fps
        (some sources are 25 fps padded to 30 like FBBench; broadcast.native_frames drops the repeats and
        label frames, which are source frame numbers, map through native_index)
        data/fb20k/clips/<match>_<k>.json  {hits: [{frame, end, hitter, type, subtype, area, point,
                                             actions, characteristics, comment, quality}], fps, offset}
usage: python tools/fb20k_prep.py [match ...]   (no args = every downloaded non-benchmark match)"""
import glob, json, os, sys

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
D = os.path.join(ROOT, "data", "fb20k")
PAD = 10
OUT_SIZE = (1280, 720)  # FBBench resolution; calibration and pixel thresholds assume it


def bench_sources():
    return {l.strip()[:4] for l in open(os.path.join(D, "benchmark_source_videos.txt")) if l.strip()}


def prep(match):
    ann = json.load(open(os.path.join(D, "ann", match + "_updated.json"), encoding="utf8"))
    video = os.path.join(D, "videos", match + ".mp4")
    out = os.path.join(D, "clips")
    os.makedirs(out, exist_ok=True)
    jobs = []
    for k, r in enumerate(ann):
        if not r.get("hitting") or "start_frame" not in r or "end_frame" not in r                 or any("start_frame" not in h for h in r["hitting"]):
            continue
        a, b = max(0, r["start_frame"] - PAD), r["end_frame"] + PAD
        dst = os.path.join(out, f"{match}_{k:03d}")
        if os.path.exists(dst + ".mp4"):
            continue
        hits = [{"frame": h["start_frame"] - a, "end": h["end_frame"] - a,
                 "hitter": h["Foundational Actions Level"]["hitter"], "type": h["Foundational Actions Level"]["hit type"],
                 "subtype": h["Foundational Actions Level"].get("subtype", []),
                 "area": h["Foundational Actions Level"]["ball area"], "point": h.get("get_point", []),
                 "actions": h["Tactical Semantics Level"].get("player actions", []),
                 "characteristics": h["Tactical Semantics Level"].get("shot characteristics", []),
                 "comment": h["Decision Evaluation Level"].get("comment", ""),
                 "quality": h["Decision Evaluation Level"].get("quality")} for h in r["hitting"]]
        json.dump({"match": match, "rally": k, "fps": r.get("fps", 25.0), "offset": a, "hits": hits,
                   "evaluation": r.get("evaluations_new", {})}, open(dst + ".json", "w", encoding="utf8"),
                  ensure_ascii=False)
        jobs.append((a, b, dst + ".mp4"))
    # one sequential pass through the match (seeking long H.264 files is slow and inexact)
    jobs.sort()
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i, j = 0, 0
    writer, cur = None, None
    while j < len(jobs):
        ok, im = cap.read()
        if not ok:
            break
        while j < len(jobs) and i > jobs[j][1]:  # passed the end of a clip
            if writer is not None and cur == j:
                writer.close(); writer = None
            j += 1
        if j < len(jobs) and jobs[j][0] <= i <= jobs[j][1]:
            if writer is None or cur != j:
                writer = imageio_ffmpeg.write_frames(jobs[j][2], OUT_SIZE, fps=fps, codec="libx264",
                                                     quality=None, macro_block_size=1,
                                                     output_params=["-crf", "18", "-preset", "fast"])
                writer.send(None)
                cur = j
            if (im.shape[1], im.shape[0]) != OUT_SIZE:  # some matches are 1080p: same 720p as FBBench
                im = cv2.resize(im, OUT_SIZE, interpolation=cv2.INTER_AREA)
            writer.send(np.ascontiguousarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
            # clips may overlap: also feed later clips that started already
        i += 1
    if writer is not None:
        writer.close()
    print(match, "clips", len(jobs), flush=True)


if __name__ == "__main__":
    bench = bench_sources()
    ms = sys.argv[1:] or sorted(os.path.basename(p)[:4] for p in glob.glob(os.path.join(D, "videos", "*.mp4")))
    for m in ms:
        if m in bench:
            print("skip benchmark source", m)
            continue
        prep(m)
