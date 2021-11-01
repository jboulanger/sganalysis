#!/bin/tcsh
############################
# Script launching jobs for processing all
# files listed in table series/filelist.csv
#
# Create 1 job by file
#
# run as sbatch slurm-job.sh
# use squeue -u $USER to monitor the job
############################

#SBATCH --job-name=sganalysis
#SBATCH --array=1-20


echo "SHELL $0"
echo "TASK ID: $SLURM_ARRAY_TASK_ID"
set dst="results"
if ( ! -d "$dst" ) then
    echo "Creating destination folder $dst"
    mkdir "$dst"
endif

cd /cephfs/jeromeb/data/StressGranule/
conda activate sganalysis
cd /cephfs/jeromeb/data/StressGranule/
python sganalysis.py --file-list series/filelist.csv --index $SLURM_ARRAY_TASK_ID \
    --output-by-granules "$dst"/granules-$SLURM_ARRAY_TASK_ID.csv \
    --output-by-cells "$dst"/cells-$SLURM_ARRAY_TASK_ID.csv \
    --output-cell-contours "$dst"/conts-$SLURM_ARRAY_TASK_ID.npz \
    --output-vignette "$dst"/vignettes-$SLURM_ARRAY_TASK_ID.png
