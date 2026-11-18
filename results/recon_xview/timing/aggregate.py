"""Merge out/phase_*.json -> results.json: per clip per stage seconds, per-method sums, medians in s per video-minute."""
import json, os

import numpy as np

D = "/scratch/ciro/timing/out"
P = {k: json.load(open(os.path.join(D, f"phase_{k}.json"))) for k in "abcde" if os.path.exists(os.path.join(D, f"phase_{k}.json"))}
STAGES = {  # stage -> (phase, keys summed)
    "P_pose1280": ("a", ["P_pose1280"]),
    "S_shuttle": ("a", ["S_tracknet_resize", "S_tracknet", "S_refine"]),
    "R_racket": ("a", ["R_racket"]),
    "B1_v1_person_boxes": ("a", ["B1_v1_person_boxes"]),
    "C2_birdscourt": ("abc", ["C2_birdscourt_prep", "C2_birdscourt_detect", "C2_birdscourt_fit"]),
    "C1_monotrack_detect": ("b", ["C1_monotrack_detect"]),
    "C1_vggt1b": ("a", ["C1_vggt"]),
    "C1_v1_ransac": ("c", ["C1_v1_ransac_fit"]),
    "O_obs": ("c", ["O_build_obs"]),
    "N_student": ("c", ["N_load_feats", "N_forward"]),
    "SP_shot_prep": ("c", ["SP_shot_prep"]),
    "V1_shuttle_fit": ("c", ["V1_shuttle_fit"]),
    "MT_shuttle_fit": ("c", ["MT_shuttle_fit"]),
    "V_vimo": ("d", ["V_vimo"]),
    "G_v1_ground_correct": ("d", ["G_ground_correct"]),
    "H_v1_hits": ("d", ["H_v1_hits"]),
    "TRAM_estimate_camera": ("e", ["TRAM_estimate_camera"]),
    "TRAM_estimate_humans": ("e", ["TRAM_estimate_humans"]),
}
METHODS = {
    "Birdseye v2 (ours)": ["C2_birdscourt", "P_pose1280", "S_shuttle", "R_racket", "O_obs", "N_student"],
    "Birdseye v1": ["C1_monotrack_detect", "C1_vggt1b", "C1_v1_ransac", "P_pose1280", "B1_v1_person_boxes", "V_vimo",
                    "G_v1_ground_correct", "S_shuttle", "SP_shot_prep", "V1_shuttle_fit", "H_v1_hits"],
    "TRAM (players only)": ["TRAM_estimate_camera", "TRAM_estimate_humans"],
    "MonoTrack (shuttle only)": ["C1_monotrack_detect", "S_shuttle", "SP_shot_prep", "MT_shuttle_fit"],
}
uids = list(P["a"]["clips"])
clips = {}
for u in uids:
    vs = P["a"]["clips"][u]["video_s"]
    st = {}
    for s, (ph, keys) in STAGES.items():
        tot, ok = 0.0, True
        for k in keys:
            v = None
            for p in ph:
                if p in P and u in P[p]["clips"] and P[p]["clips"][u] and k in P[p]["clips"][u]:
                    v = P[p]["clips"][u][k]
            if not isinstance(v, (int, float)):
                ok = False
                break
            tot += v
        st[s] = round(tot, 4) if ok else None
    clips[u] = {"n_frames": P["a"]["clips"][u]["n_frames"], "video_s": vs, "stages_s": st,
                "raw": {p: P[p]["clips"].get(u) for p in P}}


def summarize(names):
    per, tot_s, missing = [], 0.0, []
    for u in uids:
        vals = [clips[u]["stages_s"][n] for n in names]
        if any(v is None for v in vals):
            missing.append(u)
            continue
        s = sum(vals)
        per.append(s / (clips[u]["video_s"] / 60))
        tot_s += s
    mins = sum(clips[u]["video_s"] for u in uids if u not in missing) / 60
    return {"median_s_per_video_min": round(float(np.median(per)), 2) if per else None,
            "total_s": round(tot_s, 1), "total_video_min": round(mins, 3),
            "pooled_s_per_video_min": round(tot_s / mins, 2) if mins else None,
            "n_clips": len(per), "clips_missing_a_stage": missing}


stage_sum = {s: summarize([s]) for s in STAGES}
methods = {m: {"stages": ss, **summarize(ss)} for m, ss in METHODS.items()}
res = {"unit": "seconds of processing per minute of video (clip seconds = frames/25); median over clips",
       "hardware": {p: P[p].get("hw") for p in P}, "model_load_s": {p: P[p].get("load_s") for p in P},
       "methods": methods, "stages": stage_sum, "clips": clips,
       "warmup": {p: P[p].get("warmup") for p in P}}
json.dump(res, open("/scratch/ciro/timing/results.json", "w"), indent=1)
for m, r in methods.items():
    print(f"{m:28s} median {r['median_s_per_video_min']} s/min  total {r['total_s']} s over {r['total_video_min']} min "
          f"(pooled {r['pooled_s_per_video_min']}) n={r['n_clips']}")
for s, r in stage_sum.items():
    print(f"  {s:24s} median {r['median_s_per_video_min']} s/min  total {r['total_s']} s n={r['n_clips']}")
