#!/bin/bash
#SBATCH --job-name=timing_abcd
#SBATCH --account=aip-xli135
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --exclude=kn001,kn159,kn007,kn101
#SBATCH --output=/scratch/ciro/timing/logs/abcd_%j.out
# Runtime benchmark phases A-D on one L40S node (see /scratch/ciro/timing/README.md). PHASES env picks a subset.
set -u
export PYTHONUTF8=1 PYTHONUNBUFFERED=1 TMPDIR=${SLURM_TMPDIR:-/tmp} HF_HUB_OFFLINE=1
PHASES=${PHASES:-ABCD}
cd /scratch/ciro/timing
nvidia-smi; lscpu | grep "Model name"
PY=/scratch/ciro/bev2_env/bin/python
log(){ echo "[$(date +%H:%M:%S)] $*"; }
if [[ $PHASES == *A* ]]; then
  log phase A
  PYTHONPATH="/project/6101776/ciro/Badminton LLM/VGGT:/scratch/ciro/timing/pylib_vggt" $PY -u phase_a.py || echo "PHASE A FAILED"
fi
if [[ $PHASES == *B* ]]; then
  log phase B
  ( module load StdEnv/2023 gcc/12.3 opencv/4.10.0
    source /scratch/ciro/birdscourt_model_venv/bin/activate
    export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcc12/opencv/4.10.0/lib/python3.11/site-packages
    python -u phase_b.py ) || echo "PHASE B FAILED"
fi
if [[ $PHASES == *C* ]]; then
  log phase C
  $PY -u phase_c.py || echo "PHASE C FAILED"
fi
if [[ $PHASES == *D* ]]; then
  log phase D
  ( set +u; export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}; source "/project/6101776/ciro/Badminton LLM/tram/tram_env.sh"
    export PYTHONPATH=/scratch/ciro/pylib_tram:${PYTHONPATH:-}
    export TRAM_DIR="/project/6101776/ciro/Badminton LLM/tram"
    cd /scratch/ciro/timing
    python -u phase_d.py 2>&1 | grep -v -i "warning\|torch.cross\|linalg\|nested\|self.trans\|b3 =\|default value" ) || echo "PHASE D FAILED"
fi
log done
