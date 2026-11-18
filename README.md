# Birdseye

<p align="center">
  <a href="https://cirozhang.github.io/Birdseye-v2/viewer/"><img alt="Open the interactive 3D demo" src="https://img.shields.io/badge/%E2%96%B6%20Live%20demo-interactive%203D%20rally%20viewer-1d8fa8?style=for-the-badge"></a>
  &nbsp;
  <a href="https://cirozhang.github.io/Birdseye-v2/"><img alt="Project page" src="https://img.shields.io/badge/Project%20page-cirozhang.github.io-555?style=for-the-badge"></a>
</p>

A whole badminton rally in 3D from one broadcast camera: every player's 24 joints, their rackets, the shuttle and
every hit, in metres on the court.

**[Try the demo](https://cirozhang.github.io/Birdseye-v2/viewer/)**: held-out broadcast rallies in 3D, played next to
the video in your browser (no install).

## How it works

One camera cannot see depth along its own rays, so a broadcast-only method has to learn which depths are plausible.
We learn them from footage that does show depth.

- **Teacher (offline, training only).** CCTV films each match with four synchronised cameras: the broadcast view, a
  fixed side view and two player cameras. A physics fit over all four (shuttle flight under gravity and drag, sparse
  hits, VIMO bodies placed by every camera, rigid rackets) turns the 1,351 training rallies into 3D labels, and a learned
  confidence keeps only the frames it can certify. No manual 3D annotation.
- **Student (what you run).** RallyPhys, a 6.6M-parameter transformer, sees only the broadcast view's 2D detections,
  turned into camera rays: no image, no VIMO, no second camera. Nine tokens per frame (four players, the shuttle,
  four rackets), attention over time and over entities, one forward pass per rally.

The only inputs at test time are four off-the-shelf 2D detectors:
[BirdsCourt](https://github.com/CiroZhang/BirdsCourt) (court and camera),
[TrackNetV3](https://github.com/qaz812345/TrackNetV3) (shuttle), YOLOv8-pose (keypoints) and YOLOv8 (racket boxes).

## Results

Checked in a camera the method never sees at test time: the side view on four held-out CCTV matches, the top drone
on nine DroneData matches. Error (cm) = distance from the predicted 3D point to that camera's ray through its own
detection; median over rallies of the per-rally median, +- half the 95% bootstrap interval. Time: seconds of
processing per minute of video on one NVIDIA L40S.

| method | params | time | CCTV player | CCTV shuttle | cov. | Drone player | Drone shuttle | cov. |
|---|---|---|---|---|---|---|---|---|
| MonoTrack (2022) | 0 | 658 | - | 246 +- 14 | 0.76 | - | 133 +- 8 | 0.76 |
| FreeTrack (2026) | 0 | 506 | - | 232 +- 16 | 0.77 | - | 159 +- 13 | 0.76 |
| TRAM (2024) | 698M | 709 | 64.1 +- 3.5 | - | - | 33.8 +- 0.9 | - | - |
| GVHMR (2024) | 718M | 376 | 157 +- 5 | - | - | 32.8 +- 1.1 | - | - |
| **Birdseye, zero-shot** | 6.6M | **361** | 11.4 +- 0.3 | 24.3 +- 0.9 | 1.00 | **23.8** +- 0.6 | 76.2 +- 3.2 | 1.00 |
| **Birdseye + DroneData** | 6.6M | **361** | **10.1** +- 0.3 | **18.8** +- 1.2 | 1.00 | 24.4 +- 0.6 | **33.0** +- 2.1 | 1.00 |

The held-out CCTV matches come from the same two tournaments as the training matches (new players and rallies, known
venue and cameras). DroneData is a new venue and viewpoint; "zero-shot" uses no DroneData labels at all.
Every number, the command that produced it and its output files are in
[`results/recon_xview/README.md`](results/recon_xview/README.md).

## Layout

```
birdseye_v2/            the library
  court.py                court model, Camera (projection, rays)
  rally_obs.py            detections -> per-rally ray observations (the student's only input)
  rallyphys.py            the student (RallyPhys) and its losses
  rallynet.py             shared blocks and constants
  vcam.py                 virtual cameras (zero-shot training)
  teacher.py, teacher_dense.py, hitgraph.py   the multi-view physics teacher (training only)
  shuttle3d.py, pipeline.py, rallyseg.py      single-view shuttle fits (per-rally physics fit, shot segmentation)
  monotrack_fit.py, freetrack_fit.py          baselines on our inputs
  players.py, tracks.py, footlock.py          pose I/O, tracking, foot contact
  poseprior.py            pose prior
  twoview.py, refine2d.py two-drone sync and triangulation, sub-pixel shuttle refinement (DroneData ground truth)
  camtrack.py, courtfit.py, broadcast.py      per-frame broadcast cameras
  tracknet.py             lean TrackNetV3 inference
  hitnet.py, hitclf.py, v1action.py, fbfeat.py  hit and stroke models used by the experiments
tools/                  one script per stage, indexed by paper section in tools/README.md
scripts/killarney/      cluster jobs: zero-shot training (zs.slurm), GVHMR baseline
results/recon_xview/    Section 4.1 numbers, per-rally scores, logs, timing and the exact code that scored them
docs/                   project page (index.html) and the 3D viewer (viewer/)
```

## Setup

```
pip install -r requirements.txt
git clone https://github.com/CiroZhang/BirdsCourt  third_party/BirdsCourt    # 0f79b85
git clone https://github.com/qaz812345/TrackNetV3   third_party/TrackNetV3   # 6eda442
git clone https://github.com/yufu-wang/tram         third_party/tram         # 4861c11, VIMO teacher only
git clone https://github.com/Alexanderyk6/FreeTrack third_party/FreeTrack    # baseline (its 3D step is ported in birdseye_v2/freetrack_fit.py)
```

- YOLOv8 weights go in `third_party/weights/` (`yolov8x-pose.pt`, `yolov8x.pt`).
- TRAM's checkpoint and the SMPL model come from their own sites, under their own licences. They are only needed to regenerate the training-time teacher.
- BirdsCourt runs in its own Linux environment (`tools/birdscourt_wsl.py` is called through WSL).
- The BFMD stroke benchmark uses the first [Birdseye](https://github.com/CiroZhang/Birdseye) code in `third_party/birdseye_v1`.

Data and outputs are not in git:

- `data/videos/<match>/<rally>/{1.mp4 (top), (1).mp4 (back)}`: DroneData (`tools/download_videos.py`);
- `data/datacsv/`: DroneData labels;
- `data/bfmd/clips`, `data/fb20k/clips`, `data/fbbench/...`: broadcast clips;
- `data/cctv/<tournament>/...`: CCTV multi-angle matches (four feeds per match);
- `outputs/`: every intermediate result.

The GVHMR baseline runs from its own repository and environment (`scripts/killarney/run_gvhmr_2p.py`; it needs the SMPL-X model from smpl-x.is.tue.mpg.de).


## Stages

The full index is [`tools/README.md`](tools/README.md); the main path is below.

**1. Data and cameras**

| Stage | Script |
|---|---|
| CCTV matches: download, scan, cut synced rallies, split by match | `tools/cctv_fetch.py`, `tools/cctv_scan.py`, `tools/cctv_rallies.py`, `tools/cctv_split.py` |
| CCTV cameras: broadcast (BirdsCourt per rally), side view, player cameras | `tools/fb_birdscourt.py`, `tools/cctv_sidecal.py`, `tools/pcam_from_clicks.py`, `tools/pcam_calib.py` |
| DroneData: download, back camera, relabelled top-view shuttle, two-drone 3D | `tools/download_videos.py`, `tools/drone_birdscourt.py`, `tools/build_gt.py`, `tools/triangulate_poses.py` |
| Broadcast sets (BFMD, FineBadminton-20K) | `tools/fb20k_prep.py`, `tools/fb_birdscourt.py all`, `tools/bfmd_calib_fill.py`, `tools/fb_camtrack.py` |
| 2D detections | `tools/run_tracknet.py`, `tools/run_pose.py`, `tools/run_racket.py`, `tools/fb_track.py` (`POSE_IMGSZ=1280 POSE_DIR=pose1280`) |

**2. Teacher (training only)**

| Stage | Script |
|---|---|
| two-view shuttle and sync | `tools/cctv_tri.py` |
| physics fit over all cameras | `tools/cctv_teach.py` |
| which frames to trust | `tools/teach_calib.py`, `tools/teach_conf_fit.py`, `tools/teach_export.py` |
| bodies and rackets | `tools/fb_vimo.py` (TRAM environment), `tools/body_mv.py`, `tools/racket_mv.py` |
| hit detector | `tools/hit_clips.py`, `tools/hit_video_train.py`, `tools/hit_labels2.py`, `tools/hit_labels3.py` |

**3. Student**

| Stage | Script |
|---|---|
| rays | `tools/build_obs.py cctv\|drone\|bfmd\|fb\|fb20k` |
| hit labels, pose prior | `tools/build_hit_labels.py`, `tools/train_poseprior.py` |
| train / evaluate / infer | `tools/train_rallynet.py` with `ARCH=rallyphys` |

The final model (`t3`) is the last of a chain of 4,000-step fine-tunes (learning rate 1e-4) from a 20,000-step base
(`k6z`). Its stage runs with

```
ARCH=rallyphys SETS=bfmd+fb+fb20k+cctv DRONE_TRAIN=BD26+BD27+BD28+BD29+BD30 RK_DETACH=obs CONTACT3D=20 RACKET_TOP=100 \
STUDENT=1 TEACH=10 TEACH_DIR=teach_v3 XVIEW=0.05 SHUTTLE3D=20 BODY24=20 TMOT=3000 FREE_S=1 VTEACH=0 B24_SHAPE=5 \
FOOTV=60 FOOT=10 TCON=20 LR=1e-4 STEPS=4000 INIT=<previous stage> TAG=t3 python tools/train_rallynet.py train
```

The zero-shot model is the same recipe with `DRONE_DEV=0` (no DroneData labels) and virtual cameras (`VCAM=0.5`), two
stages from `k6z`: [`scripts/killarney/zs.slurm`](scripts/killarney/zs.slurm).

**4. Evaluation**

| Paper | Script |
|---|---|
| 4.1 reconstruction, CCTV side view | `tools/xview_eval.py baselines\|student\|freetrack\|gvhmr` |
| 4.1 reconstruction, DroneData top view | `XTOP=1 DRONE_CALIB=calib_bc tools/eval_shuttle.py ...`, `tools/xview_eval.py drone_student\|drone_tram\|drone_gvhmr` |
| 4.2 stroke classification | `tools/bfmd_stroke.py` (+ `tools/fbs_export.py`, `tools/stroke_baselines.py`, `tools/bfmd_rgb_stroke.py`) |
| 4.3 question answering | `tools/qa_build.py`, `tools/qa_train.py`, `tools/qa_run.py` |
| viewer data | `tools/export_rallynet_viewer.py` |

## Credits

[BirdsCourt](https://github.com/CiroZhang/BirdsCourt) · [Birdseye](https://github.com/CiroZhang/Birdseye) · [TrackNetV3](https://github.com/qaz812345/TrackNetV3) · [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · [TRAM / VIMO](https://github.com/yufu-wang/tram) · [DroneData](https://github.com/Ning-D/Drone_BD_ControlArea) · BFMD · FineBadminton

The citation will be added with the paper.
