"""Build relabeled shuttle annotations + 3D shuttle ground truth for DroneData rallies.

Per rally:
  1. TrackNetV3 2D tracks in both views (tools/run_tracknet.py), refined to the motion-blur
     centroid (birdseye_v2.refine2d).
  2. Back-minus-top time offset from two-view ray consistency (birdseye_v2.twoview).
  3. Per-frame two-view triangulation; a point is kept as 3D GT only if it viewing rays pass
     within MAX_GAP_M of each other (a TrackNet false positive in either view breaks this).

Outputs:
  outputs/gt/<match>/<rally>.json                  offset, sync stats, 3D points
  data/relabeled/<match>/<rally>/1.csv             original top-view CSV with the shuttle
      columns replaced + provenance columns:
        shuttle_x, shuttle_y   new top-view label (-1 if none)
        shuttle_src            2view = confirmed by the back view (3D consistent)
                               top    = refined TrackNet top-view detection, unconfirmed
                               none   = no detection
        shuttle_X/Y/Z          3D position in metres (court frame) for 2view rows, else empty
        old_shuttle_x/y        the original annotation, kept for comparison
usage: python tools/build_gt.py BD12 BD13 ...   (no args = all with tracks + calib)"""
import csv, glob, json, os, sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from birdseye_v2.court import Camera
from birdseye_v2.refine2d import refine_track
from birdseye_v2.twoview import estimate_offset, load_track, offset_from_hits, triangulate_track

MAX_GAP_M = 0.2  # two-view ray gap gate (see twoview.triangulate_track)
VIDEOS = os.path.join(ROOT, "data", "videos")
CSV_DIR = os.path.join(ROOT, "data", "datacsv")
TRACKS = os.path.join(ROOT, "outputs", "tracknet")
CALIB = os.path.join(ROOT, "outputs", "calib")
OUT_GT = os.path.join(ROOT, "outputs", "gt")
REFINED = os.path.join(ROOT, "outputs", "refined")  # cache: refinement streams 4K video (slow)
OUT_CSV = os.path.join(ROOT, "data", "relabeled")


def refined_tracks(m, r):
    cache = os.path.join(REFINED, m, r + ".json")
    if os.path.exists(cache):
        d = json.load(open(cache))
    else:
        d = {}
        for view, vid in (("top", "1.mp4"), ("back", "(1).mp4")):
            tr = load_track(os.path.join(TRACKS, m, r, view + ".csv"))
            px, ok, _ = refine_track(os.path.join(VIDEOS, m, r, vid), tr)
            d[view] = {"frames": tr[0][ok].tolist(), "px": np.round(px[ok], 2).tolist(), "n_det": int(len(tr[0]))}
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        json.dump(d, open(cache, "w"))
    return {v: (np.array(d[v]["frames"], int), np.array(d[v]["px"], float).reshape(-1, 2), d[v]["n_det"])
            for v in ("top", "back")}


def hit_frames(m, r):
    src = os.path.join(CSV_DIR, m, r, "1.csv")
    if not os.path.exists(src):
        return []
    return [int(x["frame"]) for x in csv.DictReader(open(src)) if x["status"] not in ("0", "")]


def sync_candidates(m, r, top, back, T, B):
    """Per-rally timing evidence: ray-consistency offset + one offset per annotated hit."""
    sync = estimate_offset(top, back, T, B)
    if not sync:
        return None
    _, per = offset_from_hits(T, B, hit_frames(m, r), guess=sync["offset"])
    sync["ray_offset"] = sync["offset"]
    sync["hit_offsets"] = [round(x, 3) for x in per]
    return sync


def match_offset(all_hits):
    """Clips of one match share (nearly) one offset: mode of the pooled per-hit offsets, refined
    as the median of the hits within 1.5 frames of it. None if too little evidence."""
    v = np.array(all_hits, float)
    if len(v) < 8:
        return None
    hist, edges = np.histogram(v, bins=np.arange(-40, 40.01, 0.25))
    mode = edges[np.argmax(hist)] + 0.125
    near = v[np.abs(v - mode) <= 1.5]
    return float(np.median(near)) if len(near) >= 5 else None


def choose_offset(sync, m_off):
    hits = np.array(sync["hit_offsets"], float)
    own = None
    if len(hits) >= 3:
        med = np.median(hits)
        agree = hits[np.abs(hits - med) <= 0.5]
        if len(agree) >= 3:
            own = float(np.median(agree))
    if own is not None and (m_off is None or abs(own - m_off) > 1.0):
        return own, "rally_hits"
    if m_off is not None:
        return m_off, "match_hits"
    return sync["ray_offset"], "rays"


def process(m, r, m_off):
    calib = json.load(open(os.path.join(CALIB, m, r + ".json")))
    top, back = Camera.from_dict(calib["top"]), Camera.from_dict(calib["back"])
    tr = refined_tracks(m, r)
    T, B = tr["top"][:2], tr["back"][:2]
    sync = sync_candidates(m, r, top, back, T, B)
    if sync:
        sync["offset"], sync["source"] = choose_offset(sync, m_off)
        sync["match_offset"] = m_off
    pts = triangulate_track(top, back, T, B, sync["offset"], max_gap_m=MAX_GAP_M) if sync else []
    gt = {"match": m, "rally": r, "sync": sync, "n_top_det": tr["top"][2], "n_top_refined": int(len(T[0])),
          "n_back_det": tr["back"][2], "n_back_refined": int(len(B[0])), "max_gap_m": MAX_GAP_M,
          "points": pts,
          "back_track": {"frames": B[0].tolist(), "px": np.round(B[1], 1).tolist()},
          "top_track": {"frames": T[0].tolist(), "px": np.round(T[1], 1).tolist()}}
    os.makedirs(os.path.join(OUT_GT, m), exist_ok=True)
    json.dump(gt, open(os.path.join(OUT_GT, m, r + ".json"), "w"))

    # relabeled top-view csv
    src = os.path.join(CSV_DIR, m, r, "1.csv")
    if os.path.exists(src):
        p3 = {p["frame"]: p["X"] for p in pts}
        top_lut = dict(zip(T[0].tolist(), T[1]))
        rows = list(csv.DictReader(open(src)))
        fields = [k for k in rows[0].keys() if k not in ("shuttle_x", "shuttle_y", "status")]
        fields += ["shuttle_x", "shuttle_y", "status", "shuttle_src", "shuttle_X", "shuttle_Y", "shuttle_Z",
                   "old_shuttle_x", "old_shuttle_y"]
        os.makedirs(os.path.join(OUT_CSV, m, r), exist_ok=True)
        with open(os.path.join(OUT_CSV, m, r, "1.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            for row in rows:
                f = int(row["frame"])
                row["old_shuttle_x"], row["old_shuttle_y"] = row["shuttle_x"], row["shuttle_y"]
                if f in top_lut:
                    row["shuttle_x"], row["shuttle_y"] = "%.1f" % top_lut[f][0], "%.1f" % top_lut[f][1]
                    row["shuttle_src"] = "2view" if f in p3 else "top"
                else:
                    row["shuttle_x"] = row["shuttle_y"] = "-1"
                    row["shuttle_src"] = "none"
                X = p3.get(f)
                row["shuttle_X"], row["shuttle_Y"], row["shuttle_Z"] = (("%.3f" % X[0], "%.3f" % X[1], "%.3f" % X[2])
                                                                       if X is not None else ("", "", ""))
                w.writerow(row)
    return gt


def _refine_job(mr):
    try:
        refined_tracks(*mr)
    except Exception as e:
        print("FAILED refine", mr, repr(e), flush=True)


def main(matches):
    # refinement streams 4K video: do it for every rally in parallel first (cached on disk)
    from concurrent.futures import ProcessPoolExecutor
    todo = [(m, os.path.basename(c)[:-5]) for m in matches for c in glob.glob(os.path.join(CALIB, m, "Rally-*.json"))
            if all(os.path.exists(os.path.join(TRACKS, m, os.path.basename(c)[:-5], v + ".csv")) for v in ("top", "back"))
            and not os.path.exists(os.path.join(REFINED, m, os.path.basename(c)[:-5] + ".json"))]
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(_refine_job, todo))
    for m in matches:
        rallies = []
        for cj in sorted(glob.glob(os.path.join(CALIB, m, "Rally-*.json"))):
            r = os.path.basename(cj)[:-5]
            if all(os.path.exists(os.path.join(TRACKS, m, r, v + ".csv")) for v in ("top", "back")):
                rallies.append(r)
        # pass 1: timing evidence from every rally of the match (refinement is cached)
        pooled = []
        for r in rallies:
            try:
                calib = json.load(open(os.path.join(CALIB, m, r + ".json")))
                tr = refined_tracks(m, r)
                s = sync_candidates(m, r, Camera.from_dict(calib["top"]), Camera.from_dict(calib["back"]),
                                    tr["top"][:2], tr["back"][:2])
                if s:
                    pooled += s["hit_offsets"]
            except Exception as e:
                print("FAILED pass1", m, r, repr(e), flush=True)
        m_off = match_offset(pooled)
        print(m, "match offset", m_off, "from", len(pooled), "hits", flush=True)
        # pass 2: triangulate with the chosen offsets
        for r in rallies:
            try:
                gt = process(m, r, m_off)
            except Exception as e:
                print("FAILED", m, r, repr(e), flush=True)
                continue
            s = gt["sync"] or {}
            print(m, r, "offset %.2f (%s) ray %.2f hits %s" % (s.get("offset", np.nan), s.get("source"),
                  s.get("ray_offset", np.nan), s.get("hit_offsets")),
                  "3d", len(gt["points"]), "/", gt["n_top_refined"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(os.listdir(CALIB)))
