#!/bin/bash

#$ -N itag2
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 1

# Array job: -t <range>
#$ -t 1

#$ -l h=node06

#$ -v DISPLAY

module load anaconda/3
source activate /var/tmp/group5/
python test.py
