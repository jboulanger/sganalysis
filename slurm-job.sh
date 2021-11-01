#!/bin/tcsh
############################
# Script launching jobs for processing all
# files listed in table series/filelist.csv
#
# Create 1 job by file
#
# run as sbatch slurm-job.sh
#
############################

#SBATCH --job-name=sganalysis
#SBATCH --array=1-20

echo "SHELL $0"
echo "TASK ID: $SLURM_ARRAY_TASK_ID"
cd /cephfs/jeromeb/data/StressGranule/
conda activate sganalysis
cd /cephfs/jeromeb/data/StressGranule/
python sganalysis.py --file-list series/filelist.csv --index $SLURM_ARRAY_TASK_ID \
    --output-by-granules granules-$SLURM_ARRAY_TASK_ID.csv \
    --output-by-cells cells-$SLURM_ARRAY_TASK_ID.csv \
    --output-cell-contours conts-$SLURM_ARRAY_TASK_ID.npz \
    --output-vignette vignettes-$SLURM_ARRAY_TASK_ID.png
