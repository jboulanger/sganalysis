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

#SBATCH --job-name=sganalysiswf-scan
#SBATCH --time=01:00:00

echo "Working directory: `pwd`"
echo "Hostname         : `hostname`"
echo "Starting time    : `date`"
echo "Data location    : $1"

# activate the python environment (may need to point to path)
conda activate sganalysis

# Download the script
if ( ! -f sganalysiswf.py ) then
    if ( ! -f ./mutex.txt ) then
        touch ./mutex.txt
        echo "Downloading script"
        wget https://raw.githubusercontent.com/jboulanger/sganalysis/master/sganalysiswf.py
        rm ./mutex.txt
    else
        echo "Downloading script in a parallel job"
        sleep 5s
    endif
else
    echo "Script already installed"
endif

python sganalysiswf.py scan \
    --file-list $2/filelist.csv \
    --data-path $2 \

echo "Finishing time   : `date`"