# Results kept for reproduction (paper Sections 4.1 and 4.2)

Snapshot taken 2026-11-17. Everything here is a copy; the working copies live under `outputs/`.

## 4.1 Cross-view reconstruction table (tab:rally)

Every method sees only the broadcast view (CCTV) or the back drone (DroneData). It is scored in a view it never sees
at test time: the fixed side camera on the held-out CCTV match, the top drone on DroneData BD15-BD23.
Error = distance (m) from the predicted 3D point to that view's ray through its own detection.
Per rally: the median over frames. Table: the median over rallies, with a 95% bootstrap interval (2000 resamples, seed 0).

| file | table cell |
|---|---|
| `cctv/cctv_mt.json` | MonoTrack, CCTV shuttle |
| `cctv/cctv_v1.json` | Birdseye v1, CCTV shuttle |
| `cctv/cctv_v2fit.json` | per-rally physics fit, CCTV shuttle |
| `cctv/cctv_tram.json` | TRAM (its own placement, VIMO `*_raw`), CCTV players |
| `cctv/cctv_v1p.json` | Birdseye v1 players (VIMO + ankle ground correction), CCTV players |
| `cctv/cctv_t3.json` | Birdseye v2 final (t3), CCTV shuttle + players |
| `drone/drone_shuttle_baselines_xtop.json` | MonoTrack (`mt`), v1 (`v1`), physics fit (`v2`), DroneData shuttle |
| `drone/drone_tram.json`, `drone/drone_v1p.json` | TRAM / v1 players, DroneData |
| `drone/drone_t3.json` | Birdseye v2 final, DroneData shuttle + players |

Format: `{rally: {shuttle, shuttle_cov, n_sh, player, player_cov, n_pl}}`. The drone baselines file is
`eval_shuttle.py` format: `rallies[i].methods[m].err` (per-point errors) and `coverage`.

Numbers at snapshot time (median [95% CI], coverage):

| | CCTV player | CCTV shuttle | DroneData player | DroneData shuttle |
|---|---|---|---|---|
| MonoTrack | - | 2.460 [2.323, 2.587], 0.76 | - | 1.326 [1.252, 1.408], 0.76 |
| TRAM | 0.641 [0.609, 0.679], 0.85 | - | 0.338 [0.328, 0.346], 0.99 | - |
| Birdseye v1 | 0.139 [0.135, 0.143], 0.86 | 1.963 [1.739, 2.175], 0.52 | 0.234 [0.229, 0.239], 0.99 | 0.694 [0.621, 0.779], 0.74 |
| physics fit (ours, no learning) | - | 1.630 [1.466, 1.788], 0.99 | - | 0.607 [0.570, 0.673], 0.89 |
| Birdseye v2 (final, t3) | 0.101 [0.098, 0.103], 0.93 | 0.188 [0.176, 0.200], 1.00 | 0.244 [0.239, 0.252], 0.99 | 0.330 [0.315, 0.356], 1.00 |

Still pending at snapshot time: Birdseye v2 without DroneData labels (Killarney zs1/zs2), GVHMR (needs SMPL-X), and
runtime (Killarney /scratch/ciro/timing).

### Commands (from the repo root, Windows Git Bash, PYTHONUTF8=1)
```
python tools/xview_eval.py baselines 8                      # CCTV: mt, v1, v2fit, tram, v1p
REDO_ERRORS=1 python tools/xview_eval.py baselines 8        # redo rallies that failed (keeps the rest)
ARCH=rallyphys STUDENT=1 BODY24=20 B24_SHAPE=5 FREE_S=1 TAG=t3 python tools/xview_eval.py student
XTOP=1 DRONE_CALIB=calib_bc python tools/eval_shuttle.py --hits detected --methods v1,v2,mt --workers 8 --tag xtop_base BD15 BD16 BD17 BD18 BD19 BD20 BD21 BD22 BD23
python tools/xview_eval.py drone_tram
ARCH=rallyphys DRONE_TRAIN=BD26+BD27+BD28+BD29+BD30 STUDENT=1 TEACH=10 TEACH_DIR=teach_v3 XVIEW=0.05 SHUTTLE3D=20 BODY24=20 TMOT=3000 FREE_S=1 VTEACH=0 B24_SHAPE=5 FOOTV=60 FOOT=10 TCON=20 TAG=t3 python tools/xview_eval.py drone_student
```
The code used is copied in `code/` (`xview_eval.py`; `eval_shuttle.py` with its XTOP mode; `cctv_tri.py` for the side
camera and frame shift). Logs are in `logs/`.

### Inputs the scores depend on
- Side camera: `outputs/cctv/side_cam*.json` + `outputs/cctv/tri/<uid>.json` (180-degree turn, frame shift).
- Side detections: `outputs/cctv_side/refined` (shuttle), `outputs/cctv_side/pose_side` (YOLOv8-pose).
- REF=tri (default): only side shuttle detections whose ray meets a TV detection's ray within 10 cm.
- Players: side people whose ankles cast onto the court (+-0.3 m), matched per frame to the method's players by ray
  distance (Hungarian, <= 2.5 m); singles and doubles.
- DroneData: top camera `outputs/calib_bc/<m>/<r>.json["top"]`; shuttle reference = the refined top-view TrackNet
  track `outputs/gt/<m>/<r>.json["top_track"]` (our relabel; the original CSV shuttle labels are not used), checked by
  eye on 24 random crops (24/24 correct); back time = top frame + sync offset. Players: DroneData CSV player positions
  cast to the floor with the top camera (tools/eval_players.gt_players), matched with tools/eval_players.match_err.
- All baselines use the BirdsCourt calibration (`DRONE_CALIB=calib_bc`), the same one as the student.

## 4.2 Stroke classification, FineBadminton-20K (`stroke/`)
- Valid rows: `fbs_t3fillown_w32_refit.json` (3D Birdseye, 83.5/81.4), `fbs_vimoown_w32_refit_fixed.json`
  (3D VIMO bodies, 83.4/81.3), `fbs_2dsh2d_w32_refit.json` (2D, 82.8/80.2), `fbs_rgbscratch_w32_refit.json`
  (CNN RGB, 69.3/63.0).
- INVALID, kept only for the record: `fbs_t3own_*` (bodies from the pre-frame-fix export) and `*_fullsh` /
  unsuffixed `fbs_vimoown_w32_refit`, `fbs_vimo_w32_refit` (older inputs).
- VideoMAE-B on FineBadminton-20K: 66.5 +- 1.1 acc, 60.4 +- 1.7 macro-F1 (seeds 0-2: 65.7/58.9, 66.1/60.1, 67.8/62.3),
  logs `stroke/fbs_rgb_videomae_refit_s*.log`.

## Runtime (timing/)
One Killarney node: NVIDIA L40S 46 GB, Intel Xeon Gold 6338, 16 CPUs (job 7740216). Same 20 held-out CCTV rallies
(3.67 min of video), warm-up rally d1026_0281 not counted, models loaded once (not timed), decoding not timed.
Seconds of processing per minute of video, median over rallies:
- Birdseye v2 (ours): 361 (BirdsCourt 147 CPU, racket YOLO 87, pose 53, TrackNet 41, obs 24, network 7)
- MonoTrack: 605 (shot prep 209, its fit 196, court detector 137, TrackNet 41)
- Birdseye v1: 2380 (camera RANSAC with VGGT seed 1157, VIMO 427, shot prep 209, fit 103, pose 53, person boxes 43)
- TRAM full pipeline (its released estimate_camera.py + estimate_humans.py, default args, per clip; job 7741636,
  `phase_e.tsv`, `phase_e.sbatch`, `tram_e_7741636.out`): 709 (camera 457, humans 230), after subtracting the fixed
  per-call cost of 77.4 s measured on a 16-frame clip (model loading). Without the subtraction: 1241. Median over 19
  clips: d1026_0277 crashed in TRAM's own estimate_camera (debug job 7745838: `UnboundLocalError: mask` in
  lib/pipeline/tools.py detect_segment_track, TRAM fails when its detector finds no person in the first frame). TRAM's time includes writing frames to disk.
Files: results.json (per clip per stage), phase_*.json, the phase scripts, aggregate.py, run_abcd.slurm, the job log.
Note: CPU fits run single-threaded (OMP_NUM_THREADS=1, the tools' default).

## 2026-11-18: Birdseye v1 dropped
Birdseye v1 was rejected, so this paper is now simply "Birdseye" and v1 is no longer a baseline. The v1 files
(`cctv_v1*.json`, `drone_v1p.json`, `v1` in the drone baselines file, v1 timing stages) are kept for the record only.

## Zero-shot row (2026-11-18)
Model `zs2` (Killarney jobs 7740707 zs1 from k6z, 7740708 zs2 from zs1; `zs.slurm`: DRONE_DEV=0, VCAM=0.5, no DroneData
labels; checkpoint copied to `outputs/rallynet/zs2.pt`, md5 f609ec4f2bd8b2dfc5b649acfcc24509).
Files `cctv/cctv_zs2.json`, `drone/drone_zs2.json`, logs `logs/log_*_zs2.txt`. Same commands as t3 with TAG=zs2.
CCTV shuttle 24.3 [23.5, 25.3] cm, player 11.4 [11.1, 11.7]; DroneData shuttle 76.2 [74.5, 80.9], player 23.8 [23.2, 24.4].
Table now reports +- half the 95% interval width. Params (from checkpoints): TRAM VIMO 697.8M; GVHMR 40.9M + HMR2.0 677.2M = 718M (ViTPose-H 637.2M is its 2D keypoint detector, excluded like our YOLOv8-pose);
  Birdseye 6.6M; MonoTrack 0.

## GVHMR row (2026-11-18)
GVHMR (SIGGRAPH Asia 2024) run on Killarney with `code/run_gvhmr_2p.py` (wrapper around its demo: its own YOLO tracking
and ViTPose/HMR2 features, static camera, K from our BirdsCourt calibration; its camera-frame SMPL-24 joints mapped
to the court with our calibration = its own placement, like TRAM `*_raw`). Jobs: CCTV 7760376, DroneData 7762316
(7760377 failed: container frame count > decodable frames; fixed by using the decoded count), timing 7760378 (L40S).
Scored with `python tools/xview_eval.py gvhmr` / `drone_gvhmr` on outputs/gvhmr/{cctv,drone} (ankles only, made by
`code/gvhmr_extract.py`). CCTV player 157.0 [152.6, 161.7] cm, cov 0.69; DroneData player 32.8 [31.7, 33.8], cov 0.99.
CCTV diagnosis: ankles median 0.62 m above floor, GVHMR distance = 0.88 x floor-contact distance (p10-p90 0.84-0.93):
a consistent depth under-estimate on the 864x486 broadcast; same K as TRAM (checked).
Time: 236 s/min own pipeline + BirdsCourt per clip = 376 s/min median (20 clips, `gvhmr/timing/*.npz` hold timings).

## FreeTrack row (2026-11-18)
FreeTrack (Machine Vision and Applications 2026, doi 10.1007/s00138-026-01929-3, github.com/Alexanderyk6/FreeTrack,
cloned to third_party/FreeTrack). Its 3D step (road/position.py + hit/move.py ground projection point) is ported in
`code/freetrack_fit.py` and fed the same track, YOLOv8-pose people and detected hits as MonoTrack (its own YOLOv12 /
MMPose / XGBoost hit detector are not used). Commands: `python tools/xview_eval.py freetrack 12`;
`XTOP=1 DRONE_CALIB=calib_bc python tools/eval_shuttle.py --hits detected --methods ft,mt --workers 12 --tag xtop_ft BD15..BD23`.
CCTV shuttle 231.6 [214.4, 245.6] cm, cov 0.77 (387 rallies, 1 error); DroneData 158.5 [148.8, 174.2], cov 0.76.
Time: 506 s/min = measured stages BirdsCourt + TrackNet + pose + shot prep (its plane step is negligible).
MonoTrack time corrected 605 -> 658 s/min: its fit uses the pose detector (foot_at), which was not counted before.
