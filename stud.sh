#!/bin/bash

#SBATCH -J halloumi          	# Name of the job
#SBATCH -N 1                	# Number of nodes
#SBATCH -n 1			        # One Task actually
#SBATCH --mem-per-cpu=4G    	# Memory allocated per CPU
#SBATCH -t 6               	    # Time allocated for the task
#SBATCH -A project02559     	# Project identification
#SBATCH --gres=gpu:h100:1   	# Desired GPU and their amount

# Redirect stdout and stderr
#SBATCH -o out/srun/%x-%j.out
#SBATCH -e out/srun/%x-%j.out


export PYTHONUNBUFFERED=1
export STDOUT_LINE_BUFFERED=1

echo "Loading modules..."
module purge
ml gcc/13 cuda/12.6
echo "\n"

start=$(date +%s)

echo "Running simulation..."
./build/halloumi

end=$(date +%s)

echo "Started at $(date -d @${start})"
echo "Ended at $(date -d @${end})"
echo "Duration: $((($end - $start) / 60)) minutes $((($end - $start) % 60)) seconds"