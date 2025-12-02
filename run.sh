#!/bin/bash
#SBATCH --array=0-9
#SBATCH --time=01:00:00
#SBATCH --mem=4000M
#SBATCH --job-name=hpstr
#SBATCH --account=hps
#SBATCH --partition=roma
#SBATCH --output=/sdf/scratch/users/r/rodwyer1/job.%A_%a.stdout

echo $SLURM_PROCID
which python3
source /sdf/group/hps/src/slic/install/bin/slic-env.sh
source /sdf/group/hps/users/rodwyer1/run/diplacedAcceptanceAsZ/hps-mc/install/bin/hps-mc-env.sh
hps-mc-job run -d $PWD/scratch/$(($SLURM_ARRAY_TASK_ID+1)) -c .hpsmc simp jobs.json -i $(($SLURM_ARRAY_TASK_ID+1))
cd /sdf/group/hps/users/rodwyer1/run/diplacedAcceptanceAsZ/hps-mc/examples/simp/output/HPS-v2019-3pt7GeV/mrhod_60/recon
wait $!
source /sdf/group/hps/users/rodwyer1/sw/workinghpstr/install/bin/hpstr-env.sh
hpstr /sdf/group/hps/users/rodwyer1/sw/workinghpstr/processors/config/recoTuple_cfg.py -i simp_3pt7_recon_$(($SLURM_ARRAY_TASK_ID+1)).slcio -o out_$(($SLURM_ARRAY_TASK_ID+1)).root
