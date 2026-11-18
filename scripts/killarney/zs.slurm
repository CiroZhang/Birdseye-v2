#!/bin/bash
#SBATCH --job-name=zs
#SBATCH --account=aip-xli135
#SBATCH --partition=gpubase_l40s_b2,gpubase_h100_b2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=7:00:00
#SBATCH --exclude=kn001,kn159,kn007,kn101
#SBATCH --output=/scratch/ciro/birdseye_v2/logs/zs_%x_%j.out
# Birdseye v2 with NO DroneData labels: the final recipe (t3: FOOTV=60 FOOT=10 TCON=20), DRONE_DEV=0, clean pose prior,
# virtual cameras VCAM=0.5, from k6z (never saw DroneData). Two 4000-step stages (zs1 from k6z, zs2 from zs1) ~ t3's h1+t3 stages.
# usage: sbatch --export=ALL,T=zs1,I=k6z zs.slurm
cd /scratch/ciro/birdseye_v2/one
export PYTHONUTF8=1 NPROC=16 OMP_NUM_THREADS=1
PY=/home/ciro/envs/bev2_env/bin/python
B="STUDENT=1 TEACH=10 TEACH_DIR=teach_v3 XVIEW=0.05 SHUTTLE3D=20 BODY24=20 TMOT=3000 FREE_S=1 VTEACH=0 B24_SHAPE=5 FOOTV=60 FOOT=10 TCON=20 PRIOR_FILE=poseprior_train.pt"
BASE="ARCH=rallyphys DRONE_TRAIN=BD26+BD27+BD28+BD29+BD30 RK_DETACH=obs CONTACT3D=20 RACKET_TOP=100 SETS=bfmd+fb+fb20k+cctv DRONE_DEV=0"
echo "$T from $I"
[ -f outputs/rallynet/$T.pt ] || env $BASE $B VCAM=0.5 INIT=$I LR=1e-4 STEPS=4000 SAVE_EVERY=500 HOLDOUT=outputs/jobs/viewer_uids.txt TAG=$T $PY -u tools/train_rallynet.py train > outputs/rallyphys_$T.raw 2>&1; grep -E "INIT|EVAL" outputs/rallyphys_$T.raw | tail -2
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES= env ARCH=rallyphys DRONE_TRAIN=BD26+BD27+BD28+BD29+BD30 $B TAG=$T $PY -u outputs/jobs/peval.py ${T}_drone > outputs/peval_${T}_drone.txt 2>&1; tail -1 outputs/peval_${T}_drone.txt
CUDA_VISIBLE_DEVICES= env ARCH=rallyphys DRONE_TRAIN=BD26+BD27+BD28+BD29+BD30 $B EVAL_SET=cctv TAG=$T $PY -u outputs/jobs/peval.py ${T}_cctv > outputs/peval_${T}_cctv.txt 2>&1; tail -1 outputs/peval_${T}_cctv.txt
echo DONE
