# tools/

One script per stage. Each starts with a docstring that says what it reads, what it writes and how to run it.
Scripts import each other by module name, so they stay in one flat folder; this index groups them by paper section.

## Data

| | |
|---|---|
| CCTV multi-angle matches | `cctv_fetch.py` (download), `cctv_probe.py`, `cctv_scan.py` (thumbnails + audio onsets), `cctv_rallies.py` (cut synced rallies from the four feeds), `cctv_audio.py`, `cctv_split.py` (train/test by match), `cctv_ingest.sh` |
| DroneData | `download_videos.py`, `drive_list.py`, `drone_birdscourt.py` (back camera; paints the yellow tape white for BirdsCourt), `build_gt.py` (relabelled top-view shuttle + two-drone 3D), `triangulate_poses.py`, `triangulate_rackets.py`, `calibrate_drag.py` |
| Broadcast sets (BFMD, FineBadminton-20K, FBBench) | `fb20k_prep.py`, `fb_birdscourt.py`, `bfmd_calib_fill.py`, `fb_camtrack.py`, `export_calib_review.py` |

## 2D detectors (shared by every method)

`run_tracknet.py`, `run_pose.py`, `run_pose_top.py`, `run_racket.py`, `fb_track.py`, `bfmd_racket.py`, `top_racket.py`,
`birdscourt_wsl.py` (BirdsCourt inside WSL).

## Section 3.3: multi-view teacher (training only)

| | |
|---|---|
| cameras | `cctv_sidecal.py` (side view), `pcam_from_clicks.py` + `cctv_pcam.py` + `pcam_calib.py` (pan-tilt-zoom player cameras), `pcam_eval.py` |
| shuttle | `cctv_tri.py` (TV + side triangulation, sync), `pcam_shuttle.py`, `pcam_streak.py` (motion-blur streaks in close-ups), `cctv_teach.py` (the dense physics fit, `birdseye_v2/teacher_dense.py`) |
| which frames to trust | `teach_calib.py` (self-hiding tests), `teach_conf_fit.py` (learned confidence), `teach_eval.py`, `teach_export.py` |
| bodies and rackets | `mv_people.py`, `mv_track.py`, `mv_wrist.py`, `body_mv.py`, `racket3d.py`, `racket_mv.py`, `fb_vimo.py`, `drone_vimo.py` (VIMO, TRAM environment) |
| hit detector | `audio_hits.py`, `hit_clips.py`, `hit_video_train.py`, `hit_labels2.py`, `hit_labels3.py`, `hit_pick.py`, `hit_decode.py`, `self_round.py` |

## Sections 3.4-3.5: broadcast-only student

`build_obs.py` (detections -> rays, the network's only input), `build_hit_labels.py`, `train_poseprior.py`,
`train_rallynet.py` (`ARCH=rallyphys`; training, evaluation and inference), `smpl_dechumpy.py`.
The recipe of the released runs is in `scripts/killarney/zs.slurm` (zero-shot) and the top-level README.

## Section 4.1: 3D rally reconstruction

| | |
|---|---|
| cross-view scoring | `xview_eval.py` (CCTV side view; stages `baselines`, `student`, `freetrack`, `gvhmr`, `drone_student`, `drone_tram`, `drone_gvhmr`), `eval_shuttle.py` (DroneData, `XTOP=1`), `eval_players.py` |
| baselines | MonoTrack and FreeTrack in `birdseye_v2/monotrack_fit.py`, `birdseye_v2/freetrack_fit.py`; TRAM through `drone_vimo.py`/`fb_vimo.py`; GVHMR through `scripts/killarney/run_gvhmr_2p.py` + `gvhmr_extract.py` |
| other checks | `eval_racket.py`, `hit_f1.py`, `hit_eval_fb.py`, `cctv_baselines.py` |

The numbers, the commands that made them and every output file are in `results/recon_xview/README.md`.

## Section 4.2: stroke classification

`bfmd_stroke.py` (classifier and protocol), `bfmd_export.py`, `bfmd_export_2d.py`, `rallynet_bfmd_export.py`,
`fbs_export.py`, `stroke3d.py`, `stroke_baselines.py` (BST, TemPose), `bfmd_rgb_stroke.py` + `bfmd_frames.py` (RGB),
`hit3d.py`, `bfmd_ensemble.py`, `v1_finetune.py`, `fb_label_map.py`, `fb_build_train.py`.

## Section 4.3: question answering

`qa_build.py`, `qa_run.py`, `qa_scene.py`, `qa_frames.py`, `qa_train.py`, `qa2_build.py`, `qa2_common.py`,
`qa2_train.py`, `qa2_eval.py`, `fbqa_build.py`, `fbbench_answer.py`, `fb_scoreboard.py`, `fb_score_chain.py`,
`fb_board_names.py`.

## Viewers

`export_viewer3d.py` (the 3D rally viewer in `docs/viewer`), `export_rallynet_viewer.py` (the teacher-vs-student viewer in `docs/teacher`), `export_viewer.py`, `export_demo.py`, `export_fb_viewer.py`,
`fb_full_example.py`.

## Earlier experiments (kept for the record, not used by the paper)

`calibrate_all.py`, `fb_calibrate.py` (replaced by BirdsCourt), `shot_fit.py`, `shot_fit_sp.py`, `racket_test.py`,
`train_hitclf.py`, `train_hitnet.py`, `tvpose_crops.py`, `tvpose_data.py`, `tvpose_train.py`.
