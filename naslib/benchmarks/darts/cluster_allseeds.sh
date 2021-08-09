#!/bin/bash
#SBATCH -q dlc-krishnan
#SBATCH -p mlhiwidlc_gpu-rtx2080 
#SBATCH --gres=gpu:1          # reserves one GPU
#SBATCH -o logs/naslib_darts_%x.%A-%a.%N.out       # STDOUT  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -e logs/naslib_darts_%x.%A-%a.%N.err       # STDERR  %A will be replaced by the SLURM_ARRAY_JOB_ID value
#SBATCH -a 1-5 # array size
#SBATCH -J NSLB_DARTS_V2_EVAL

echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

start=`date +%s`

# Activate virtual env so that run_experiment can load the correct packages
#source /home/zelaa/NASLib/nl-venv/bin/activate
source /home/krishnan/miniconda3/bin/activate
conda activate naslib

#--config-file $1 --optimizer $2 evaluation.batch_size ${3:-96}
python runner.py --seed $SLURM_ARRAY_TASK_ID


end=`date +%s`
runtime=$((end-start))

echo Runtime: $runtime
