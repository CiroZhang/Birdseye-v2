#!/bin/bash
#SBATCH --job-name=timing_tram
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=05:00:00
#SBATCH --exclude=kn001,kn159,kn007,kn101
#SBATCH --output=/scratch/ciro/timing/logs/tram_e_%j.out
# TRAM's own full pipeline (estimate_camera.py + estimate_humans.py, default args) per clip on the 20 timing clips;
# first line of uids_e.txt is the warm-up. A 16-frame clip measures the fixed per-call cost (model loading).
nvidia-smi -L
source /scratch/ciro/timing/tram/env.sh
CL=/scratch/ciro/birdseye_v2/repo/data/cctv/clips
OUT=/scratch/ciro/timing/out/phase_e.tsv; echo -e "uid\tnframes\tcamera_s\thumans_s" > $OUT
mkdir -p /tmp/te; ffmpeg -loglevel error -y -i $CL/$(sed -n 2p /scratch/ciro/timing/uids_e.txt).mp4 -frames:v 16 -c:v libx264 /tmp/te/short16.mp4
for v in /tmp/te/short16.mp4 $(sed "s|^|$CL/|; s|$|.mp4|" /scratch/ciro/timing/uids_e.txt); do
  s=$(basename $v .mp4); rm -rf results/$s
  t0=$(date +%s.%N); python scripts/estimate_camera.py --video $v > /dev/null 2>&1 || echo "CAM_FAIL $s"
  t1=$(date +%s.%N); python scripts/estimate_humans.py --video $v > /dev/null 2>&1 || echo "HUM_FAIL $s"
  t2=$(date +%s.%N)
  echo -e "$s\t$(ls results/$s/images | wc -l)\t$(echo "$t1-$t0" | bc)\t$(echo "$t2-$t1" | bc)" | tee -a $OUT
done
echo DONE
