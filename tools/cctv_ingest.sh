#!/usr/bin/env bash
# One CCTV session, raw downloads -> training data, no manual step:
#   scan (thumbnails + audio onsets) -> audio sync + rally cut (main camera found automatically) -> TV clips into the
#   "cctv" set, side clips into "cctv_side" -> TV: shuttle, pose, court calibration, rackets, VIMO bodies -> rally
#   observations -> side: camera calibration (once) + shuttle -> two-view triangulation (frame shift / court frame
#   found automatically) -> train/test split by match -> two-view physics teacher.
# GPU stages wait until no training (train_rallynet.py) is running and outputs/gpu_reserved is absent (an
# experiment queued for the GPU): one GPU job at a time.
# usage: bash tools/cctv_ingest.sh <session> tv=<id> side=<id> [p1=<id> p2=<id>]   (videos in data/cctv/<session>/<id>.mp4)
set -u
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
S=$1; shift
D=data/cctv/$S
L=outputs/cctv_ingest_$S
mkdir -p $L
log() { echo "[$(date +%H:%M)] $S $*"; }
gpu_free() { while [ -f outputs/gpu_reserved ]; do sleep 120; done; while powershell -NoProfile -c "if (Get-CimInstance Win32_Process | ? { \$_.CommandLine -match 'train_rallynet[.]py train' }) { exit 1 }" >/dev/null 2>&1; [ $? -ne 0 ]; do sleep 120; done; }
fails() { echo "FAILED $(grep -c FAILED $1)"; }
declare -A V; for a in "$@"; do V[${a%%=*}]=${a#*=}; done

ids=""; for k in "${!V[@]}"; do ids="$ids $D/${V[$k]}.mp4"; done
todo=""; for p in $ids; do [ -f "${p%.mp4}.thumbs.npy" ] && [ -f "${p%.mp4}.onset.npy" ] || todo="$todo $p"; done
[ -n "$todo" ] && { log scan; python -u tools/cctv_scan.py $todo > $L/scan.txt 2>&1; }
[ -f $D/rallies/index.json ] && [ "$(find $D/rallies -name '*.mp4' | wc -l)" -ge "$(( $(python -c "import json;print(len(json.load(open('$D/rallies/index.json'))['rallies']))") * ${#V[@]} ))" ] || {
  log rallies; python -u tools/cctv_rallies.py $D "$@" > $L/rallies.txt 2>&1; grep -E "main camera|rallies," $L/rallies.txt; }
python - <<EOF
import glob, os
for view, d in (("tv", "data/cctv/clips"), ("side", "data/cctv/clips_side"), ("p1", "data/cctv/clips_p1"), ("p2", "data/cctv/clips_p2")):
    os.makedirs(d, exist_ok=True)
    n = 0
    for p in glob.glob("$D/rallies/*/" + view + ".mp4"):
        dst = os.path.join(d, os.path.basename(os.path.dirname(p)) + ".mp4")
        if not os.path.exists(dst):
            os.link(p, dst); n += 1
    print(view, n, "new clips")
EOF
gpu_free; log shuttle tv; FB_SET=cctv SHUTTLE_ONLY=1 python -u tools/fb_track.py > $L/track.txt 2>&1; fails $L/track.txt
gpu_free; log pose tv; FB_SET=cctv POSE_ONLY=1 POSE_IMGSZ=1280 POSE_DIR=pose1280 python -u tools/fb_track.py > $L/pose.txt 2>&1; fails $L/pose.txt
gpu_free; log calib; FB_SET=cctv python -u tools/fb_birdscourt.py all > $L/calib.txt 2>&1; grep -E "^fit|^courtfit" $L/calib.txt
gpu_free; log racket; FB_SET=cctv IMGSZ=1280 python -u tools/bfmd_racket.py > $L/racket.txt 2>&1; fails $L/racket.txt
gpu_free; log vimo; FB_SET=cctv POSE_DIR=pose1280 .venv_vimo/Scripts/python -u tools/fb_vimo.py > $L/vimo.txt 2>&1; fails $L/vimo.txt
log obs; python -u tools/build_obs.py cctv > $L/obs.txt 2>&1; fails $L/obs.txt
log side calib; python -u tools/cctv_sidecal.py $S
gpu_free; log shuttle side; FB_SET=cctv_side SHUTTLE_ONLY=1 python -u tools/fb_track.py > $L/side_track.txt 2>&1; fails $L/side_track.txt
log tri; python -u tools/cctv_tri.py $(ls data/cctv/clips_side | grep "^${S}_" | sed 's/\.mp4$//') > $L/tri.txt 2>&1; cat $L/tri.txt
log split; python -u tools/cctv_split.py
log teach; CUDA_VISIBLE_DEVICES=-1 python -u tools/cctv_teach.py $(ls data/cctv/clips_side | grep "^${S}_" | sed 's/\.mp4$//') > $L/teach.txt 2>&1; fails $L/teach.txt
log INGEST_DONE
