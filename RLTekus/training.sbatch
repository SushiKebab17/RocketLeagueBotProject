#!/bin/bash
#
#SBATCH --job-name=TekusTestAdvObs256x64x128  # Job name for tracking
#SBATCH --partition=cpu-batch       # Partition you wish to use (see above for list)
#SBATCH --cpus-per-task=40          # Number of CPU threads used by your job, set this upto 40 as required.
#SBATCH --mem-per-cpu=1500          # RAM in MB needed per core (i.e. 1500MB * 40 threads = 60GB RAM)
#SBATCH --time=2-00:00:00           # Job time limit: D-HH:MM:SS
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80 # Events to send email on, remove if you don't want this
#SBATCH --output=/JobOutputs/joboutput_%j.out # Standard out from your job
#SBATCH --error=/JobOutputs/joboutput_%j.err  # Standard error from your job

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

## Execute your program(s) ##
srun python3.9 training.py
## Other examples:
## A Python program
# srun python3 my_program.py
#
## Excutable program (e.g. with shebang or written in C/C++)
# srun ~/myprogram