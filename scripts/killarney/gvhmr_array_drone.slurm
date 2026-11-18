#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --partition=gpubase_l40s_b2,gpubase_h100_b2
#SBATCH --exclude=kn001,kn159,kn007,kn101
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=gvhmr_drone
#SBATCH --array=0-28
#SBATCH --output=/scratch/ciro/gvhmr/logs/drone_%A_%a.out
# 288 DroneData test rallies (BD15-BD23, back drone), 10 per task; calib = BirdsCourt back camera (calib_bc).
source /scratch/ciro/gvhmr/env.sh
source $G/venv/bin/activate
PER=10
UIDS=$(sed -n "$((SLURM_ARRAY_TASK_ID*PER+1)),$((SLURM_ARRAY_TASK_ID*PER+PER))p" $G/drone_uids.txt)
echo "task $SLURM_ARRAY_TASK_ID: $UIDS"; nvidia-smi -L
[ -n "$UIDS" ] && python $G/run_gvhmr_2p.py --uids $UIDS --clips $G/drone_clips --calib $G/drone_calib --prep $G/prep_drone --out $G/out_drone "$@"
