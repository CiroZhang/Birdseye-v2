#!/bin/bash
#SBATCH --account=aip-xli135
#SBATCH --partition=gpubase_l40s_b2,gpubase_h100_b2
#SBATCH --exclude=kn001,kn159,kn007
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=gvhmr_cctv
#SBATCH --array=0-39
#SBATCH --output=/scratch/ciro/gvhmr/logs/arr_%A_%a.out
# 393 test uids, 10 per task. Re-submittable: finished npz are skipped, cached prep is reused.
source /scratch/ciro/gvhmr/env.sh
source $G/venv/bin/activate
PER=10
UIDS=$(sed -n "$((SLURM_ARRAY_TASK_ID*PER+1)),$((SLURM_ARRAY_TASK_ID*PER+PER))p" $G/test_uids.txt)
echo "task $SLURM_ARRAY_TASK_ID: $UIDS"; nvidia-smi -L
[ -n "$UIDS" ] && python $G/run_gvhmr_2p.py --uids $UIDS "$@"
