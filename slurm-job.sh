#!/bin/tcsh
############################
# Script launching jobs for processing all
# files listed in table series/filelist.csv
#
# Create 1 job by file
#
# run as sbatch slurm-job.sh [csv file] [index] [data path]
# use squeue -u $USER to monitor the job
############################

#SBATCH --job-name=sganalysis
#SBATCH --time=01:00:00
echo "Working directory: `pwd`"
echo "Hostname         : `hostname`"
echo "Starting time    : `date`"
echo "Array task id    : $SLURM_ARRAY_TASK_ID"
echo "Input file       : $1"
@ line = ( $SLURM_ARRAY_TASK_ID + 1 )
set row = `sed -n {$line}p $1`
echo "Row              : $row"
echo "Data location    : $3"

# activate the python environment (may need to point to path)
conda activate sganalysis

# Download the script
if ( ! -f sganalysis.py ) then
    echo "Downloading script"
    wget https://raw.githubusercontent.com/jboulanger/sganalysis/master/sganalysis.py
endif

set dst = $3/results
if ( ! -d "$dst" ) then
    echo "Creating destination folder $dst"
    mkdir "$dst"
endif

python sganalysis.py z
    --file-list $1 \
    --data-path $3 \
    --index $SLURM_ARRAY_TASK_ID  \
    --output-by-granules "$dst"/granules-$SLURM_ARRAY_TASK_ID.csv \
    --output-by-cells "$dst"/cells-$SLURM_ARRAY_TASK_ID.csv \
    --output-cell-contours "$dst"/conts-$SLURM_ARRAY_TASK_ID.npz \
    --output-vignette "$dst"/vignettes-$SLURM_ARRAY_TASK_ID.png

echo "Finishing time   : `date`"
