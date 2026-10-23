# Birdseye v2

A whole badminton rally in 3D from one camera: players, rackets, the shuttle and every hit.

The only inputs are four off-the-shelf 2D detectors:

- court and camera from [BirdsCourt](https://github.com/CiroZhang/BirdsCourt);
- shuttle from [TrackNetV3](https://github.com/qaz812345/TrackNetV3);
- player keypoints from YOLOv8-pose;
- racket boxes from YOLOv8.

Everything from there to 3D is this repository. Every detection becomes a ray in court coordinates. A small transformer reads the whole rally. It alternates attention across time and across entities (players, shuttle), and it drives a physical model of the rally:

- **Skeleton.** The network outputs joint angles, and forward kinematics with bone lengths fixed per player turns them into joints. A body is held up only while a foot is on the floor.
- **Racket.** A rigid racket set by an angle in the forearm's frame. Which hand holds it is learned once per player for the rally. A racket detection only counts for a hand that can reach it.
- **Hits.** Who hits, and when.
- **Shuttle.** Between hits it flies under gravity and quadratic drag (terminal speed 6.4 m/s, measured from two-view flights). At a hit it is at the hitter's racket head.

Training uses broadcast rallies with no 3D labels, plus:

- hit labels where a dataset has them;
- poses triangulated from two drones on separate DroneData matches;
- VIMO bone directions, during training only.

At run time the model needs no VIMO, no SMPL and no second camera.

**Project page and interactive viewer:** `docs/` (GitHub Pages). Locally, run `python -m http.server -d docs 8000` and open http://localhost:8000.

## Layout

```
birdseye_v2/          the library
  court.py              court model, Camera (projection, rays)
  rally_obs.py          detections -> per-rally ray observations (the model's only input)
  rallyphys.py          the model and its training objective (E_obs + E_force + E_prior)
  rallynet.py           shared blocks, constants and the earlier direct-regression baseline
  shuttle3d.py          gravity + drag flights, joint multi-shot fit
  pipeline.py           shuttle reconstruction around the fit
  players.py, tracks.py pose I/O, player tracking, racket boxes
  poseprior.py          pose prior (autoencoder over 12 body joints)
  twoview.py, refine2d.py  two-drone sync and triangulation, sub-pixel shuttle refinement (ground truth)
  camtrack.py, courtfit.py, broadcast.py  per-frame broadcast cameras (cuts, pans, zooms), native frame rates
  tracknet.py           lean TrackNetV3 inference
tools/                one script per stage (each starts with its usage)
docs/                 project page (index.html) and the 3D viewer (viewer/)
```

## Setup

```
pip install -r requirements.txt
git clone https://github.com/CiroZhang/BirdsCourt  third_party/BirdsCourt    # 0f79b85
git clone https://github.com/qaz812345/TrackNetV3   third_party/TrackNetV3   # 6eda442
git clone https://github.com/yufu-wang/tram         third_party/tram         # 4861c11, VIMO teacher only
```

- YOLOv8 weights go in `third_party/weights/` (`yolov8x-pose.pt`, `yolov8x.pt`).
- TRAM's checkpoint and the SMPL model come from their own sites, under their own licences. They are only needed to regenerate the training-time teacher.
- BirdsCourt runs in its own Linux environment (`tools/birdscourt_wsl.py` is called through WSL).
- The BFMD stroke benchmark uses the first [Birdseye](https://github.com/CiroZhang/Birdseye) code in `third_party/birdseye_v1`.

Data and outputs are not in git:

- `data/videos/<match>/<rally>/{1.mp4 (top), (1).mp4 (back)}`: DroneData (`tools/download_videos.py`);
- `data/datacsv/`: DroneData labels;
- `data/bfmd/clips`, `data/fb20k/clips`, `data/fbbench/...`: broadcast clips;
- `outputs/`: every intermediate result.

## Stages

**DroneData (evaluation, plus dev matches BD12–14 in training)**

| Stage | Script |
|---|---|
| download | `tools/download_videos.py BD12 ...` |
| back camera | `tools/drone_birdscourt.py BD12 ...` |
| 2D detections, back view | `tools/run_tracknet.py`, `tools/run_pose.py`, `tools/run_racket.py` |
| ground truth (top view, never an input) | `tools/build_gt.py` (two-view shuttle), `tools/run_pose_top.py` + `tools/triangulate_poses.py` (3D poses) |
| drag constant | `tools/calibrate_drag.py` |

**Broadcast training sets (BFMD, FineBadminton)**

| Stage | Script |
|---|---|
| clips | `tools/fb20k_prep.py` |
| cameras | `tools/fb_birdscourt.py all`, `tools/bfmd_calib_fill.py`, `tools/fb_camtrack.py` |
| 2D detections | `tools/fb_track.py` (`POSE_IMGSZ=1280 POSE_DIR=pose1280`), `tools/bfmd_racket.py` |
| VIMO teacher (training only) | `tools/fb_vimo.py`, `tools/drone_vimo.py` (TRAM environment) |

**Model**

| Stage | Script |
|---|---|
| rays | `tools/build_obs.py drone\|bfmd\|fb\|fb20k` |
| hit labels | `tools/build_hit_labels.py` |
| pose prior | `tools/train_poseprior.py` |
| train and score | `TAG=p11 tools/train_rallynet.py train` (the defaults reproduce the released model) |
| score only | `TAG=p11 tools/train_rallynet.py eval` (`FULL=1` for every held-out rally) |
| viewer data | `TAG=p11 VIEWER_OUT=../docs/viewer tools/export_rallynet_viewer.py BD15/Rally-39 ...` |

**Representation test (BFMD strokes)**

| Stage | Script |
|---|---|
| export bodies in Birdseye v1's layout | `TAG=p11 tools/rallynet_bfmd_export.py` |
| train and score the stroke model | `PROTOCOL=clean tools/bfmd_stroke.py` |
| VIMO baseline | `tools/bfmd_export.py` |

## Credits

[BirdsCourt](https://github.com/CiroZhang/BirdsCourt) · [Birdseye](https://github.com/CiroZhang/Birdseye) · [TrackNetV3](https://github.com/qaz812345/TrackNetV3) · [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) · [TRAM / VIMO](https://github.com/yufu-wang/tram) · [DroneData](https://github.com/Ning-D/Drone_BD_ControlArea) · BFMD · FineBadminton

The citation will be added with the paper.
